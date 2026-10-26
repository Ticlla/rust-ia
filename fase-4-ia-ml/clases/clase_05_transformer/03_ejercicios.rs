//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 05: Arquitectura Transformer y Memoria KV-Cache
//! ARCHIVO: 03_ejercicios.rs
//! ==============================================================================
//!
//! EJERCICIOS PRÁCTICOS Y SUITE DE TESTS:
//! 1. Scaled Dot-Product Attention con máscara causal opcional.
//! 2. Estructura KV-Cache con gestión de memoria y concatenación de secuencia.
//! 3. Capa completa MultiHeadAttention (MHA) compatible con Prefill y Decode autorregresivo.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

/// Crea una máscara causal triangular inferior con -inf en el triángulo superior estricto.
pub fn create_causal_mask(seq_len_q: usize, seq_len_k: usize, dev: &Device) -> Result<Tensor> {
    let mut mask_data = Vec::with_capacity(seq_len_q * seq_len_k);
    for i in 0..seq_len_q {
        for j in 0..seq_len_k {
            if j > i {
                mask_data.push(f32::NEG_INFINITY);
            } else {
                mask_data.push(0.0f32);
            }
        }
    }
    Tensor::from_vec(mask_data, (1, 1, seq_len_q, seq_len_k), dev)
}

/// Ejercicio 1: Scaled Dot-Product Attention
/// Attention(Q, K, V) = softmax((Q @ K^T) / sqrt(d_k) + mask) @ V
///
/// q: [Batch, Heads, Seq_Q, Dim]
/// k: [Batch, Heads, Seq_K, Dim]
/// v: [Batch, Heads, Seq_K, Dim]
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    is_causal: bool,
) -> Result<Tensor> {
    let (_, _, seq_len_q, head_dim) = q.dims4()?;
    let (_, _, seq_len_k, _) = k.dims4()?;

    // 1. Q @ K^T -> [Batch, Heads, Seq_Q, Seq_K]
    let k_t = k.transpose(2, 3)?;
    let scale = (head_dim as f64).sqrt();
    let scores = (q.matmul(&k_t)? / scale)?;

    // 2. Máscara Causal si se requiere y seq_len_q > 1
    let scores = if is_causal && seq_len_q > 1 {
        let mask = create_causal_mask(seq_len_q, seq_len_k, q.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // 3. Normalización Softmax a lo largo del eje de llaves (dimensión 3)
    let weights = candle_nn::ops::softmax(&scores, 3)?;

    // 4. Proyección de valores: Weights @ V -> [Batch, Heads, Seq_Q, Dim]
    weights.matmul(v)
}

/// Ejercicio 2: KV-Cache para generación autorregresiva O(1)
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Concatena los tensores del nuevo token a lo largo del eje temporal (dimensión 2).
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
            _ => candle_core::bail!("Estado inválido de KV-Cache: k o v desincronizados"),
        }
    }

    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(t) => t.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_none() || self.seq_len() == 0
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Ejercicio 3: Capa Multi-Head Attention completa con soporte para KV-Cache
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
        assert_eq!(embed_dim % n_heads, 0, "embed_dim debe ser divisible por n_heads");
        let head_dim = embed_dim / n_heads;

        // Inicializamos pesos deterministas o aleatorios normalizados
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

    /// Forward pass de Multi-Head Attention:
    /// x: [Batch, Seq_len, Embed_dim]
    /// cache: KV-Cache opcional para reutilizar keys y values de pasos anteriores
    pub fn forward(&self, x: &Tensor, cache: &mut Option<KvCache>) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.dims3()?;

        // 1. Proyecciones Lineales
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape a múltiples cabezas: [Batch, Seq, Heads, Head_dim] -> Transpose a [Batch, Heads, Seq, Head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // 3. Gestión del KV-Cache
        let (k_used, v_used, is_causal) = match cache {
            Some(c) => {
                let (full_k, full_v) = c.append(&k, &v)?;
                // Si seq_len > 1 estamos en Prefill (requiere causal mask)
                // Si seq_len == 1 estamos en Decode (el nuevo token puede atender a todo el cache)
                let causal = seq_len > 1;
                (full_k, full_v, causal)
            }
            None => (k, v, seq_len > 1),
        };

        // 4. Scaled Dot-Product Attention
        let attn_out = scaled_dot_product_attention(&q, &k_used, &v_used, is_causal)?;

        // 5. Concatenar cabezas de vuelta: [Batch, Heads, Seq, Head_dim] -> [Batch, Seq, Embed_dim]
        let attn_out = attn_out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch_size, seq_len, self.embed_dim))?;

        // 6. Proyección final de salida W_o
        self.o_proj.forward(&attn_out)
    }
}

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 05: EJERCICIOS PRÁCTICOS — TRANSFORMER & KV-CACHE");
    println!("==============================================================================");

    let dev = Device::Cpu;
    let embed_dim = 16;
    let n_heads = 4;
    let mha = MultiHeadAttention::new(embed_dim, n_heads, &dev)?;

    println!("Capa MultiHeadAttention creada exitosamente:");
    println!("- Dimensión de embedding: {}", embed_dim);
    println!("- Número de cabezas: {}", n_heads);
    println!("- Dimensión por cabeza: {}", mha.head_dim);

    // Simular un pipeline completo: Prefill de 3 tokens y 2 pasos de Decode
    let mut cache = Some(KvCache::new());

    println!("\n[Paso 1: Prefill del Prompt (3 tokens)]");
    let prompt = Tensor::randn(0.0f32, 1.0f32, (1, 3, embed_dim), &dev)?;
    let out_prefill = mha.forward(&prompt, &mut cache)?;
    println!("- Salida del Prefill: {:?}", out_prefill.shape());
    println!("- Longitud actual del Cache: {}", cache.as_ref().unwrap().seq_len());

    println!("\n[Paso 2: Generación Token 4]");
    let token_4 = Tensor::randn(0.0f32, 1.0f32, (1, 1, embed_dim), &dev)?;
    let out_t4 = mha.forward(&token_4, &mut cache)?;
    println!("- Salida Token 4: {:?}", out_t4.shape());
    println!("- Longitud actual del Cache: {}", cache.as_ref().unwrap().seq_len());

    println!("\n[Paso 3: Generación Token 5]");
    let token_5 = Tensor::randn(0.0f32, 1.0f32, (1, 1, embed_dim), &dev)?;
    let out_t5 = mha.forward(&token_5, &mut cache)?;
    println!("- Salida Token 5: {:?}", out_t5.shape());
    println!("- Longitud actual del Cache: {}", cache.as_ref().unwrap().seq_len());

    println!("\nEjecuta los tests unitarios automatizados con:");
    println!("cargo test --bin c05_ejercicios");
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
    fn test_scaled_dot_product_attention_causal() -> Result<()> {
        let dev = Device::Cpu;
        let batch = 1;
        let heads = 1;
        let seq = 3;
        let dim = 2;

        let q = Tensor::ones((batch, heads, seq, dim), candle_core::DType::F32, &dev)?;
        let k = Tensor::ones((batch, heads, seq, dim), candle_core::DType::F32, &dev)?;
        // Asignamos a cada posición de V un valor identificable: [10.0, 20.0, 30.0]
        let v_data = vec![
            10.0f32, 10.0f32, // token 0
            20.0f32, 20.0f32, // token 1
            30.0f32, 30.0f32, // token 2
        ];
        let v = Tensor::from_vec(v_data, (batch, heads, seq, dim), &dev)?;

        let out = scaled_dot_product_attention(&q, &k, &v, true)?;
        let out_vals = out.squeeze(0)?.squeeze(0)?.to_vec2::<f32>()?;

        // Con máscara causal:
        // Token 0 SOLO puede atender a Token 0 -> resultado debe ser [10.0, 10.0]
        assert!(
            (out_vals[0][0] - 10.0).abs() < 1e-4,
            "El token 0 debe atender exclusivamente a sí mismo"
        );

        // Token 1 atiende a 0 y 1 (promedio de 10.0 y 20.0 = 15.0)
        assert!(
            (out_vals[1][0] - 15.0).abs() < 1e-4,
            "El token 1 debe atender a tokens 0 y 1"
        );

        // Token 2 atiende a 0, 1 y 2 (promedio de 10, 20, 30 = 20.0)
        assert!(
            (out_vals[2][0] - 20.0).abs() < 1e-4,
            "El token 2 debe atender a tokens 0, 1 y 2"
        );

        Ok(())
    }

    #[test]
    fn test_kv_cache_growth_and_shapes() -> Result<()> {
        let dev = Device::Cpu;
        let mut cache = KvCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);

        // 1. Prefill con 3 tokens
        let k_prefill = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &dev)?;
        let v_prefill = Tensor::zeros((1, 2, 3, 4), candle_core::DType::F32, &dev)?;
        let (cached_k, cached_v) = cache.append(&k_prefill, &v_prefill)?;
        assert_eq!(cached_k.dims4()?, (1, 2, 3, 4));
        assert_eq!(cached_v.dims4()?, (1, 2, 3, 4));
        assert_eq!(cache.seq_len(), 3);
        assert!(!cache.is_empty());

        // 2. Decode de 1 token adicional
        let k_step1 = Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &dev)?;
        let v_step1 = Tensor::zeros((1, 2, 1, 4), candle_core::DType::F32, &dev)?;
        let (cached_k, cached_v) = cache.append(&k_step1, &v_step1)?;
        assert_eq!(cached_k.dims4()?, (1, 2, 4, 4));
        assert_eq!(cached_v.dims4()?, (1, 2, 4, 4));
        assert_eq!(cache.seq_len(), 4);

        // 3. Reset
        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);

        Ok(())
    }

    #[test]
    fn test_multi_head_attention_forward_con_cache() -> Result<()> {
        let dev = Device::Cpu;
        let embed_dim = 8;
        let n_heads = 2;
        let mha = MultiHeadAttention::new(embed_dim, n_heads, &dev)?;

        let mut cache = Some(KvCache::new());

        // Prefill con longitud 4
        let prompt = Tensor::zeros((1, 4, embed_dim), candle_core::DType::F32, &dev)?;
        let out_prefill = mha.forward(&prompt, &mut cache)?;
        assert_eq!(out_prefill.dims3()?, (1, 4, embed_dim));
        assert_eq!(cache.as_ref().unwrap().seq_len(), 4);

        // Decode de 1 token individual
        let next_token = Tensor::zeros((1, 1, embed_dim), candle_core::DType::F32, &dev)?;
        let out_decode = mha.forward(&next_token, &mut cache)?;
        assert_eq!(out_decode.dims3()?, (1, 1, embed_dim));
        assert_eq!(cache.as_ref().unwrap().seq_len(), 5);

        Ok(())
    }
}
