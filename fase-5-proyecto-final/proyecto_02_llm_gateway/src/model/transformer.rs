//! Modelo Transformer Liviano para Inferencia con Paged KV-Cache

use crate::config::GatewayConfig;
use crate::paged_cache::PagedKvCacheManager;
use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

pub struct TransformerLayerWeights {
    pub norm1_w: Tensor,
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub norm2_w: Tensor,
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

pub struct GatewayTransformer {
    pub config: GatewayConfig,
    pub dev: Device,
    pub embedding: Embedding,
    pub layers: Vec<TransformerLayerWeights>,
    pub final_norm_w: Tensor,
    pub lm_head: Linear,
}

impl GatewayTransformer {
    pub fn new(config: GatewayConfig, dev: &Device) -> Result<Self> {
        let embed_dim = config.embed_dim;
        let vocab_size = config.vocab_size;
        let intermediate_dim = embed_dim * 2;

        let w_emb = Tensor::randn(0.0f32, 1.0f32, (vocab_size, embed_dim), dev)?;
        let embedding = Embedding::new(w_emb, embed_dim);

        let mut layers = Vec::with_capacity(config.n_layers);
        let scale = (1.0 / (embed_dim as f64).sqrt()) as f32;

        for _ in 0..config.n_layers {
            let norm1_w = Tensor::ones(embed_dim, candle_core::DType::F32, dev)?;
            let q_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?, None);
            let k_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?, None);
            let v_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?, None);
            let o_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (embed_dim, embed_dim), dev)? * (scale as f64))?, None);

            let norm2_w = Tensor::ones(embed_dim, candle_core::DType::F32, dev)?;
            let gate_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (intermediate_dim, embed_dim), dev)? * (scale as f64))?, None);
            let up_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (intermediate_dim, embed_dim), dev)? * (scale as f64))?, None);
            let down_proj = Linear::new((Tensor::randn(0.0f32, 1.0f32, (embed_dim, intermediate_dim), dev)? * (scale as f64))?, None);

            layers.push(TransformerLayerWeights {
                norm1_w,
                q_proj,
                k_proj,
                v_proj,
                o_proj,
                norm2_w,
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        let final_norm_w = Tensor::ones(embed_dim, candle_core::DType::F32, dev)?;
        let w_lm = (Tensor::randn(0.0f32, 1.0f32, (vocab_size, embed_dim), dev)? * (scale as f64))?;
        let lm_head = Linear::new(w_lm, None);

        Ok(Self {
            config,
            dev: dev.clone(),
            embedding,
            layers,
            final_norm_w,
            lm_head,
        })
    }

    fn apply_rms_norm(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let x_sq = x.sqr()?;
        let mean = x_sq.mean_keepdim(candle_core::D::Minus1)?;
        let norm = x.broadcast_div(&(mean + 1e-5)?.sqrt()?)?;
        norm.broadcast_mul(weight)
    }

    /// Ejecuta un paso de inferencia para una secuencia individual, integrando la escritura y lectura
    /// de su Paged KV-Cache
    pub fn step_sequence(
        &self,
        token_ids: &[u32],
        is_prefill: bool,
        block_table: &[usize],
        paged_cache: &mut PagedKvCacheManager,
    ) -> Result<Tensor> {
        let seq_len = token_ids.len();
        let input_tensor = Tensor::new(&token_ids[..], &self.dev)?.unsqueeze(0)?; // [1, S]
        let mut hidden = self.embedding.forward(&input_tensor)?; // [1, S, D]
        let head_dim = self.config.embed_dim / self.config.n_heads;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 1. RMSNorm 1
            let norm_x = Self::apply_rms_norm(&hidden, &layer.norm1_w)?;

            // 2. Proyecciones Q, K, V
            let q = layer.q_proj.forward(&norm_x)?;
            let k = layer.k_proj.forward(&norm_x)?;
            let v = layer.v_proj.forward(&norm_x)?;

            // Reshape para cabezas: [1, S, H, D]
            let q_heads = q.reshape((1, seq_len, self.config.n_heads, head_dim))?.transpose(1, 2)?; // [1, H, S, D]
            let k_heads = k.reshape((1, seq_len, self.config.n_heads, head_dim))?.transpose(1, 2)?;
            let v_heads = v.reshape((1, seq_len, self.config.n_heads, head_dim))?.transpose(1, 2)?;

            // 3. Escribir nuevos tokens en el Paged KV-Cache
            if is_prefill {
                for s in 0..seq_len {
                    let logical_block = s / self.config.block_size;
                    let offset_in_block = s % self.config.block_size;
                    let phys_block = block_table[logical_block];

                    let k_tok = k_heads.squeeze(0)?.narrow(1, s, 1)?.squeeze(1)?; // [H, D]
                    let v_tok = v_heads.squeeze(0)?.narrow(1, s, 1)?.squeeze(1)?; // [H, D]
                    paged_cache.write_token(layer_idx, phys_block, offset_in_block, &k_tok, &v_tok)?;
                }
            } else {
                // Decode: Solo 1 token
                let curr_pos = seq_len - 1;
                let logical_block = curr_pos / self.config.block_size;
                let offset_in_block = curr_pos % self.config.block_size;
                let phys_block = block_table[logical_block];

                let k_tok = k_heads.squeeze(0)?.narrow(1, 0, 1)?.squeeze(1)?; // [H, D]
                let v_tok = v_heads.squeeze(0)?.narrow(1, 0, 1)?.squeeze(1)?;
                paged_cache.write_token(layer_idx, phys_block, offset_in_block, &k_tok, &v_tok)?;
            }

            // 4. Gather de todo el contexto acumulado de la secuencia
            let (cached_k, cached_v) = paged_cache.gather_sequence_kv(layer_idx, block_table, seq_len)?;

            // 5. Scaled Dot-Product Attention
            let k_t = cached_k.transpose(2, 3)?;
            let scale = (head_dim as f64).sqrt();
            let scores = (q_heads.matmul(&k_t)? / scale)?;

            // Máscara Causal si estamos en prefill y S > 1
            let scores = if is_prefill && seq_len > 1 {
                let mut mask_data = Vec::with_capacity(seq_len * seq_len);
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if j > i {
                            mask_data.push(f32::NEG_INFINITY);
                        } else {
                            mask_data.push(0.0f32);
                        }
                    }
                }
                let mask = Tensor::from_vec(mask_data, (1, 1, seq_len, seq_len), &self.dev)?;
                scores.broadcast_add(&mask)?
            } else {
                scores
            };

            let weights = candle_nn::ops::softmax(&scores, 3)?;
            let attn_out = weights.matmul(&cached_v)?; // [1, H, S, D]
            let attn_out = attn_out
                .transpose(1, 2)?
                .contiguous()?
                .reshape((1, seq_len, self.config.embed_dim))?;
            let attn_proj = layer.o_proj.forward(&attn_out)?;

            // Residual 1
            hidden = (&hidden + &attn_proj)?;

            // Residual 2: RMSNorm + SwiGLU MLP
            let norm_x2 = Self::apply_rms_norm(&hidden, &layer.norm2_w)?;
            let gate = candle_nn::ops::silu(&layer.gate_proj.forward(&norm_x2)?)?;
            let up = layer.up_proj.forward(&norm_x2)?;
            let mlp_hidden = (gate * up)?;
            let mlp_out = layer.down_proj.forward(&mlp_hidden)?;

            hidden = (&hidden + &mlp_out)?;
        }

        // Final Norm y LM Head
        let final_norm = Self::apply_rms_norm(&hidden, &self.final_norm_w)?;
        let logits = self.lm_head.forward(&final_norm)?; // [1, S, Vocab_size]

        // Retornar logits del último token
        let last_logits = logits.squeeze(0)?.narrow(0, seq_len - 1, 1)?.squeeze(0)?;
        Ok(last_logits)
    }

    /// Muestreo determinista (Greedy) o con Temperatura
    pub fn sample_token(logits: &Tensor, temperature: f64) -> Result<u32> {
        if temperature <= 0.001 {
            let vals = logits.to_vec1::<f32>()?;
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &v) in vals.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }
            return Ok(max_idx as u32);
        }

        let scaled = (logits / temperature)?;
        let probs = candle_nn::ops::softmax(&scaled, 0)?;
        let vals = probs.to_vec1::<f32>()?;

        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in vals.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        Ok(max_idx as u32)
    }
}
