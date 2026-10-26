//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 06: Checkpoint Final — Servidor de Inferencia LLM con Token Streaming
//! ARCHIVO: 03_ejercicios.rs
//! ==============================================================================
//!
//! CHECKPOINT INTEGRADOR Y SUITE DE TESTS:
//! 1. Arquitectura modular de MiniLlm (Embedding + Transformer Layers + RmsNorm + LM Head).
//! 2. Motor de inferencia autorregresivo con soporte para Token Streaming mediante callbacks.
//! 3. 3 Tests unitarios rigurosos que validan sampling, forward pass y terminación por EOS.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};
use std::io::{self, Write};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

/// Configuración de hiperparámetros del LLM
#[derive(Debug, Clone)]
pub struct LlmConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub n_heads: usize,
    pub intermediate_dim: usize,
    pub n_layers: usize,
    pub eos_token_id: u32,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            vocab_size: 16,
            embed_dim: 16,
            n_heads: 4,
            intermediate_dim: 32,
            n_layers: 2,
            eos_token_id: 2,
        }
    }
}

/// Normalización RMSNorm
pub struct RmsNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(dim: usize, dev: &Device) -> Result<Self> {
        let weight = Tensor::ones(dim, candle_core::DType::F32, dev)?;
        Ok(Self { weight, eps: 1e-5 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let norm = x.broadcast_div(&(mean + self.eps)?.sqrt()?)?;
        norm.broadcast_mul(&self.weight)
    }
}

/// Capa MLP SwiGLU: SiLU(x @ W_gate) * (x @ W_up) @ W_down
pub struct SwiGluMlp {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl SwiGluMlp {
    pub fn new(embed_dim: usize, intermediate_dim: usize, dev: &Device) -> Result<Self> {
        let scale = (1.0 / (embed_dim as f64).sqrt()) as f32;
        let w_gate = (Tensor::randn(0.0f32, 1.0f32, (intermediate_dim, embed_dim), dev)? * (scale as f64))?;
        let w_up = (Tensor::randn(0.0f32, 1.0f32, (intermediate_dim, embed_dim), dev)? * (scale as f64))?;
        let w_down = (Tensor::randn(0.0f32, 1.0f32, (embed_dim, intermediate_dim), dev)? * (scale as f64))?;

        Ok(Self {
            gate_proj: Linear::new(w_gate, None),
            up_proj: Linear::new(w_up, None),
            down_proj: Linear::new(w_down, None),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// KV-Cache
#[derive(Debug, Default, Clone)]
pub struct KvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (None, None) => {
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
                Ok((new_k.clone(), new_v.clone()))
            }
            (Some(past_k), Some(past_v)) => {
                let full_k = Tensor::cat(&[past_k, new_k], 2)?;
                let full_v = Tensor::cat(&[past_v, new_v], 2)?;
                self.k = Some(full_k.clone());
                self.v = Some(full_v.clone());
                Ok((full_k, full_v))
            }
            _ => candle_core::bail!("Inconsistencia en KV-Cache"),
        }
    }

    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(t) => t.dim(2).unwrap_or(0),
            None => 0,
        }
    }
}

/// Capa Multi-Head Attention
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
    pub embed_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, n_heads: usize, dev: &Device) -> Result<Self> {
        let head_dim = embed_dim / n_heads;
        let scale = (1.0 / (embed_dim as f64).sqrt()) as f32;
        let w_q = (Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?;
        let w_k = (Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?;
        let w_v = (Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?;
        let w_o = (Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?;

        Ok(Self {
            q_proj: Linear::new(w_q, None),
            k_proj: Linear::new(w_k, None),
            v_proj: Linear::new(w_v, None),
            o_proj: Linear::new(w_o, None),
            n_heads,
            head_dim,
            embed_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, cache: &mut Option<KvCache>) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let k = self.k_proj.forward(x)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let v = self.v_proj.forward(x)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;

        let (k_used, v_used, is_causal) = match cache {
            Some(c) => {
                let (full_k, full_v) = c.append(&k, &v)?;
                (full_k, full_v, s > 1)
            }
            None => (k, v, s > 1),
        };

        let k_t = k_used.transpose(2, 3)?;
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k_t)? / scale)?;

        let scores = if is_causal {
            let (_, _, total_k, _) = k_used.dims4()?;
            let mut mask_data = Vec::with_capacity(s * total_k);
            for i in 0..s {
                for j in 0..total_k {
                    if j > i {
                        mask_data.push(f32::NEG_INFINITY);
                    } else {
                        mask_data.push(0.0f32);
                    }
                }
            }
            let mask = Tensor::from_vec(mask_data, (1, 1, s, total_k), q.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        let weights = candle_nn::ops::softmax(&scores, 3)?;
        let attn_out = weights.matmul(&v_used)?;

        let attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.embed_dim))?;

        self.o_proj.forward(&attn_out)
    }
}

/// Bloque Transformer Completo
pub struct TransformerBlock {
    pub norm1: RmsNorm,
    pub attn: MultiHeadAttention,
    pub norm2: RmsNorm,
    pub mlp: SwiGluMlp,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, n_heads: usize, intermediate_dim: usize, dev: &Device) -> Result<Self> {
        Ok(Self {
            norm1: RmsNorm::new(embed_dim, dev)?,
            attn: MultiHeadAttention::new(embed_dim, n_heads, dev)?,
            norm2: RmsNorm::new(embed_dim, dev)?,
            mlp: SwiGluMlp::new(embed_dim, intermediate_dim, dev)?,
        })
    }

    pub fn forward(&self, x: &Tensor, cache: &mut Option<KvCache>) -> Result<Tensor> {
        let norm_x1 = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&norm_x1, cache)?;
        let x = (x + attn_out)?;

        let norm_x2 = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x2)?;
        x + mlp_out
    }
}

/// Mini-LLM Completo
pub struct MiniLlm {
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RmsNorm,
    pub lm_head: Linear,
    pub config: LlmConfig,
}

impl MiniLlm {
    pub fn new(config: LlmConfig, dev: &Device) -> Result<Self> {
        let w_emb = Tensor::randn(0.0f32, 1.0f32, (config.vocab_size, config.embed_dim), dev)?;
        let embedding = Embedding::new(w_emb, config.embed_dim);

        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(TransformerBlock::new(
                config.embed_dim,
                config.n_heads,
                config.intermediate_dim,
                dev,
            )?);
        }

        let final_norm = RmsNorm::new(config.embed_dim, dev)?;
        let w_lm = Tensor::randn(0.0f32, 1.0f32, (config.vocab_size, config.embed_dim), dev)?;
        let lm_head = Linear::new(w_lm, None);

        Ok(Self {
            embedding,
            layers,
            final_norm,
            lm_head,
            config,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, caches: &mut [Option<KvCache>]) -> Result<Tensor> {
        let mut x = self.embedding.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut caches[i])?;
        }
        x = self.final_norm.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

/// Motor de Inferencia con Soporte de Token Streaming
pub struct InferenceEngine {
    pub model: MiniLlm,
    pub tokenizer: Tokenizer,
    pub dev: Device,
}

impl InferenceEngine {
    pub fn new(model: MiniLlm, tokenizer: Tokenizer, dev: Device) -> Self {
        Self { model, tokenizer, dev }
    }

    /// Muestreo determinista o probabilístico
    pub fn sample(logits: &Tensor, temperature: f64) -> Result<u32> {
        if temperature <= 0.001 {
            let vec = logits.to_vec1::<f32>()?;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in vec.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            return Ok(max_idx as u32);
        }

        let scaled = (logits / temperature)?;
        let probs = candle_nn::ops::softmax(&scaled, 0)?;
        let vec = probs.to_vec1::<f32>()?;
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in vec.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Ok(max_idx as u32)
    }

    /// Genera texto token a token invocando el callback `on_token` para streaming en vivo
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f64,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
        if prompt_ids.is_empty() {
            candle_core::bail!("El prompt no puede estar vacío");
        }

        let mut caches: Vec<Option<KvCache>> = vec![Some(KvCache::new()); self.model.config.n_layers];

        // 1. Prefill
        let input_tensor = Tensor::new(&prompt_ids[..], &self.dev)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, &mut caches)?;
        let prompt_len = prompt_ids.len();
        let last_logits = logits.squeeze(0)?.get(prompt_len - 1)?;
        let mut next_token = Self::sample(&last_logits, temperature)?;

        let mut full_output = prompt.to_string();

        // 2. Decode Loop
        for _ in 0..max_new_tokens {
            if next_token == self.model.config.eos_token_id {
                break;
            }

            let token_str = self
                .tokenizer
                .decode(&[next_token], false)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            on_token(&token_str);
            full_output.push(' ');
            full_output.push_str(&token_str);

            let single_input = Tensor::new(&[next_token], &self.dev)?.unsqueeze(0)?;
            let single_logits = self.model.forward(&single_input, &mut caches)?;
            let token_logits = single_logits.squeeze(0)?.get(0)?;
            next_token = Self::sample(&token_logits, temperature)?;
        }

        Ok(full_output)
    }
}

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 06: EJERCICIOS PRÁCTICOS — SERVIDOR DE INFERENCIA CON STREAMING");
    println!("==============================================================================");

    let dev = Device::Cpu;

    // Vocabulario y Tokenizador
    let vocab = [
        ("<unk>".to_string(), 0),
        ("<s>".to_string(), 1),
        ("</s>".to_string(), 2),
        ("Rust".to_string(), 3),
        ("potencia".to_string(), 4),
        ("la".to_string(), 5),
        ("IA".to_string(), 6),
        ("moderna".to_string(), 7),
    ];
    let wl = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(wl);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

    let config = LlmConfig {
        vocab_size: 8,
        embed_dim: 16,
        n_heads: 2,
        intermediate_dim: 32,
        n_layers: 2,
        eos_token_id: 2,
    };

    let model = MiniLlm::new(config, &dev)?;
    let engine = InferenceEngine::new(model, tokenizer, dev);

    let prompt = "Rust potencia la IA";
    println!("Prompt inicial: \"{}\"", prompt);
    print!("Streaming en vivo: {}", prompt);
    io::stdout().flush().unwrap();

    let full_text = engine.generate_stream(prompt, 5, 0.7, |chunk| {
        print!(" {}", chunk);
        io::stdout().flush().unwrap();
    })?;

    println!("\n\nTexto final completado: \"{}\"", full_text);
    println!("Para correr los tests unitarios automatizados:");
    println!("cargo test --bin c06_ejercicios");
    println!("==============================================================================");
    Ok(())
}

// ==============================================================================
// TESTS UNITARIOS AUTOMATIZADOS
// ==============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_greedy_vs_temperature() -> Result<()> {
        let dev = Device::Cpu;
        // Vector de logits donde el índice 2 es claramente el ganador: [1.0, 2.5, 9.8, 0.2]
        let logits = Tensor::from_vec(vec![1.0f32, 2.5, 9.8, 0.2], 4, &dev)?;

        // Greedy puro (T = 0.0)
        let sampled_greedy = InferenceEngine::sample(&logits, 0.0)?;
        assert_eq!(sampled_greedy, 2, "Greedy debe retornar el índice máximo (2)");

        // Temperatura normal (T = 0.5)
        let sampled_temp = InferenceEngine::sample(&logits, 0.5)?;
        assert_eq!(sampled_temp, 2, "Temperatura baja debe favorecer fuertemente al índice máximo");

        Ok(())
    }

    #[test]
    fn test_mini_llm_forward_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let config = LlmConfig {
            vocab_size: 10,
            embed_dim: 16,
            n_heads: 4,
            intermediate_dim: 32,
            n_layers: 2,
            eos_token_id: 2,
        };

        let model = MiniLlm::new(config.clone(), &dev)?;
        let mut caches: Vec<Option<KvCache>> = vec![Some(KvCache::new()); config.n_layers];

        // 1. Prefill con 3 tokens: [1, 3]
        let input_prefill = Tensor::new(&[[1u32, 3, 5]], &dev)?;
        let logits_prefill = model.forward(&input_prefill, &mut caches)?;
        assert_eq!(logits_prefill.dims3()?, (1, 3, 10));
        assert_eq!(caches[0].as_ref().unwrap().seq_len(), 3);
        assert_eq!(caches[1].as_ref().unwrap().seq_len(), 3);

        // 2. Decode con 1 token: [1, 1]
        let input_decode = Tensor::new(&[[6u32]], &dev)?;
        let logits_decode = model.forward(&input_decode, &mut caches)?;
        assert_eq!(logits_decode.dims3()?, (1, 1, 10));
        assert_eq!(caches[0].as_ref().unwrap().seq_len(), 4);
        assert_eq!(caches[1].as_ref().unwrap().seq_len(), 4);

        Ok(())
    }

    #[test]
    fn test_pipeline_generacion_completo_con_eos() -> Result<()> {
        let dev = Device::Cpu;
        let vocab = [
            ("<unk>".to_string(), 0),
            ("<s>".to_string(), 1),
            ("</s>".to_string(), 2),
            ("Hola".to_string(), 3),
            ("mundo".to_string(), 4),
        ];
        let wl = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("<unk>".to_string())
            .build()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let mut tokenizer = Tokenizer::new(wl);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        let config = LlmConfig {
            vocab_size: 5,
            embed_dim: 8,
            n_heads: 2,
            intermediate_dim: 16,
            n_layers: 1,
            eos_token_id: 2,
        };

        let model = MiniLlm::new(config, &dev)?;
        let engine = InferenceEngine::new(model, tokenizer, dev);

        let mut streamed_tokens = Vec::new();
        let full_text = engine.generate_stream("Hola mundo", 4, 0.0, |token| {
            streamed_tokens.push(token.to_string());
        })?;

        assert!(!streamed_tokens.is_empty(), "Debe haberse emitido al menos 1 token");
        assert!(full_text.starts_with("Hola mundo"));

        Ok(())
    }
}
