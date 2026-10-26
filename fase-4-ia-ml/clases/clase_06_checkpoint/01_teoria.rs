//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 06: Checkpoint Final — Servidor de Inferencia LLM con Token Streaming
//! ARCHIVO: 01_teoria.rs
//! ==============================================================================
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Integrar todos los componentes de la Fase 4: Tokenizer + Candle + KV-Cache.
//! 2. Comprender la anatomía del ciclo Prefill -> Decode autorregresivo.
//! 3. Implementar estrategias de muestreo (Greedy Argmax vs Temperatura).
//! 4. Diseñar el pipeline de Token Streaming en vivo con vaciado de buffer (flush).
//! 5. Analizar las dos métricas reinas de LLM Serving: TTFT y TPOT.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};
use std::io::{self, Write};
use std::thread::sleep;
use std::time::{Duration, Instant};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

// Importamos los conceptos construidos en las clases 04 y 05:
// - RMSNorm
// - SwiGLU
// - Scaled Dot-Product Attention con KV-Cache

/// RMSNorm pedagógico: x / sqrt(mean(x^2) + eps) * weight
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

/// SwiGLU MLP: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
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

/// KV-Cache persistente para cada capa del modelo
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
            _ => candle_core::bail!("Inconsistencia interna en el KV-Cache"),
        }
    }

    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(t) => t.dim(2).unwrap_or(0),
            None => 0,
        }
    }
}

/// Capa de Atención Multi-Cabeza con KV-Cache
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

/// Bloque Transformer Completo: Pre-RMSNorm + MHA + Pre-RMSNorm + SwiGLU MLP
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
        // Conexión Residual 1: x + Attention(RMSNorm(x))
        let norm_x1 = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&norm_x1, cache)?;
        let x = (x + attn_out)?;

        // Conexión Residual 2: x + MLP(RMSNorm(x))
        let norm_x2 = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x2)?;
        x + mlp_out
    }
}

/// Selector de muestreo: Greedy Argmax o Temperatura
pub fn sample_next_token(logits: &Tensor, temperature: f64) -> Result<u32> {
    // Si la temperatura es <= 0.0, usamos búsqueda codiciosa pura (Greedy)
    if temperature <= 0.001 {
        let logits_vec = logits.to_vec1::<f32>()?;
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &val) in logits_vec.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }

    // Muestreo con Temperatura: Softmax(logits / T)
    let scaled_logits = (logits / temperature)?;
    let probs = candle_nn::ops::softmax(&scaled_logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;

    // Muestreo categórico pseudo-aleatorio o greedy top
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in probs_vec.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    Ok(max_idx as u32)
}

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 06: CHECKPOINT FINAL — SERVIDOR DE INFERENCIA LLM CON STREAMING");
    println!("==============================================================================");

    let dev = Device::Cpu;

    // 1. Construir un Tokenizador de Vocabulario Real
    let vocab = [
        ("<unk>".to_string(), 0),
        ("<s>".to_string(), 1),
        ("</s>".to_string(), 2),
        ("Rust".to_string(), 3),
        ("es".to_string(), 4),
        ("el".to_string(), 5),
        ("futuro".to_string(), 6),
        ("de".to_string(), 7),
        ("la".to_string(), 8),
        ("inteligencia".to_string(), 9),
        ("artificial".to_string(), 10),
        ("y".to_string(), 11),
        ("sistemas".to_string(), 12),
        ("rápidos".to_string(), 13),
    ];
    let vocab_size = vocab.len();
    let word_level = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("<unk>".to_string())
        .build()
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let mut tokenizer = Tokenizer::new(word_level);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

    // 2. Parámetros del Mini-LLM
    let embed_dim = 16;
    let n_heads = 4;
    let intermediate_dim = 32;
    let n_layers = 2;

    println!("[1] Inicializando Arquitectura del LLM:");
    println!("    - Vocabulario: {} tokens", vocab_size);
    println!("    - Embedding Dim: {}", embed_dim);
    println!("    - Capas Transformer: {}", n_layers);
    println!("    - Dispositivo: {:?}", dev);

    // Inicializamos Pesos del Modelo
    let w_emb = Tensor::randn(0.0f32, 1.0f32, (vocab_size, embed_dim), &dev)?;
    let embedding = Embedding::new(w_emb, embed_dim);

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        layers.push(TransformerBlock::new(embed_dim, n_heads, intermediate_dim, &dev)?);
    }
    let final_norm = RmsNorm::new(embed_dim, &dev)?;
    let w_lm_head = Tensor::randn(0.0f32, 1.0f32, (vocab_size, embed_dim), &dev)?;
    let lm_head = Linear::new(w_lm_head, None);

    // 3. Prompt de Entrada del Usuario
    let prompt_text = "Rust es el futuro de la";
    let encoding = tokenizer.encode(prompt_text, false).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("\n[2] Prompt del Usuario: \"{}\"", prompt_text);
    println!("    Token IDs: {:?}", prompt_ids);

    // Estructuras de KV-Cache: Una por cada capa del Transformer
    let mut layer_caches: Vec<Option<KvCache>> = vec![Some(KvCache::new()); n_layers];

    // -------------------------------------------------------------------------
    // FASE PREFILL: Procesar todos los tokens del prompt juntos
    // -------------------------------------------------------------------------
    println!("\n[3] Ejecutando Fase PREFILL (Llenado del KV-Cache)...");
    let prefill_start = Instant::now();

    let input_tensor = Tensor::new(&prompt_ids[..], &dev)?.unsqueeze(0)?; // [1, S]
    let mut hidden = embedding.forward(&input_tensor)?; // [1, S, D]

    for (layer_idx, layer) in layers.iter().enumerate() {
        hidden = layer.forward(&hidden, &mut layer_caches[layer_idx])?;
    }
    hidden = final_norm.forward(&hidden)?;
    let logits = lm_head.forward(&hidden)?; // [1, S, Vocab_size]

    let ttft = prefill_start.elapsed();
    println!("    -> TTFT (Time To First Token): {:.2} ms", ttft.as_secs_f64() * 1000.0);
    println!("    -> Tokens en KV-Cache tras Prefill: {}", layer_caches[0].as_ref().unwrap().seq_len());

    // Extraer logits de la última posición del prompt para predecir el primer token nuevo
    let prompt_len = prompt_ids.len();
    let last_logits = logits.squeeze(0)?.get(prompt_len - 1)?;
    let mut next_token_id = sample_next_token(&last_logits, 0.7)?;

    // -------------------------------------------------------------------------
    // FASE DECODE CON TOKEN STREAMING EN VIVO
    // -------------------------------------------------------------------------
    println!("\n[4] Iniciando Fase DECODE con Token Streaming en Vivo:");
    print!("    Salida generada: {}", prompt_text);
    io::stdout().flush().unwrap();

    let max_new_tokens = 6;
    let eos_token_id = 2u32;
    let mut generated_tokens = Vec::new();
    let decode_start = Instant::now();

    for _step in 1..=max_new_tokens {
        if next_token_id == eos_token_id {
            print!(" <EOS>");
            io::stdout().flush().unwrap();
            break;
        }

        generated_tokens.push(next_token_id);

        // Detokenizar y emitir en tiempo real
        let token_chunk = tokenizer.decode(&[next_token_id], false).unwrap_or_default();
        print!(" {}", token_chunk);
        // CRÍTICO: Vaciar el búfer de stdout para que el usuario reciba el streaming instantáneo
        io::stdout().flush().unwrap();

        // Pequeño delay pedagógico para visualizar el streaming interactivo
        sleep(Duration::from_millis(150));

        // Preparar tensor de UN SOLO TOKEN para el siguiente paso (O(1))
        let single_input = Tensor::new(&[next_token_id], &dev)?.unsqueeze(0)?;
        let mut single_hidden = embedding.forward(&single_input)?;

        for (layer_idx, layer) in layers.iter().enumerate() {
            single_hidden = layer.forward(&single_hidden, &mut layer_caches[layer_idx])?;
        }
        single_hidden = final_norm.forward(&single_hidden)?;
        let single_logits = lm_head.forward(&single_hidden)?;

        // Obtener logits del único token procesado
        let token_logits = single_logits.squeeze(0)?.get(0)?;
        next_token_id = sample_next_token(&token_logits, 0.7)?;
    }
    println!();

    let total_decode_time = decode_start.elapsed();
    let num_gen = generated_tokens.len();
    let tpot = if num_gen > 0 {
        total_decode_time.as_secs_f64() * 1000.0 / (num_gen as f64)
    } else {
        0.0
    };

    println!("\n[5] Métricas de Producción del Servidor:");
    println!("    - Tokens Generados: {}", num_gen);
    println!("    - Longitud final en KV-Cache: {}", layer_caches[0].as_ref().unwrap().seq_len());
    println!("    - TPOT (Time Per Output Token): {:.2} ms/token", tpot);
    println!("    - Throughput: {:.2} tokens/segundo", if tpot > 0.0 { 1000.0 / tpot } else { 0.0 });
    println!("==============================================================================");
    Ok(())
}
