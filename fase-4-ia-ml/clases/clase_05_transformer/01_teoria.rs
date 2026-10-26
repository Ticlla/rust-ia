//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 05: Arquitectura Transformer y Memoria KV-Cache
//! ARCHIVO: 01_teoria.rs
//! ==============================================================================
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Entender el cuello de botella de la inferencia autorregresiva (Prefill vs Decode).
//! 2. Implementar Scaled Dot-Product Attention: softmax(Q @ K.T / sqrt(d_k) + mask) @ V.
//! 3. Comprender RoPE (Rotary Position Embeddings) y cómo preserva posiciones relativas.
//! 4. Diseñar e implementar el KV-Cache en memoria en Rust para decodificación O(1).
//! 5. Simular la generación paso a paso de un Transformer autorregresivo.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};

/// Implementación pedagógica de RoPE (Rotary Position Embeddings).
/// Rota pares adyacentes de coordenadas en el plano complejo usando frecuencias theta.
///
/// x: Tensor [batch, n_heads, seq_len, head_dim] (head_dim debe ser par)
/// offset: posición inicial en la secuencia para esta llamada (crítico para KV-Cache)
pub fn apply_rope(x: &Tensor, offset: usize) -> Result<Tensor> {
    let (b, h, seq_len, d) = x.dims4()?;
    assert!(d % 2 == 0, "head_dim debe ser divisible por 2 para RoPE");

    let mut flat = x.flatten_all()?.to_vec1::<f32>()?;
    let base = 10000.0f32;

    for b_idx in 0..b {
        for h_idx in 0..h {
            for s in 0..seq_len {
                let pos = (offset + s) as f32;
                let offset_base = ((b_idx * h + h_idx) * seq_len + s) * d;
                for i in 0..(d / 2) {
                    let theta = base.powf(-2.0 * (i as f32) / (d as f32));
                    let angle = pos * theta;
                    let cos_val = angle.cos();
                    let sin_val = angle.sin();

                    let idx0 = offset_base + 2 * i;
                    let idx1 = offset_base + 2 * i + 1;

                    let x0 = flat[idx0];
                    let x1 = flat[idx1];

                    flat[idx0] = x0 * cos_val - x1 * sin_val;
                    flat[idx1] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }

    Tensor::from_vec(flat, (b, h, seq_len, d), x.device())
}

/// Crea una máscara causal triangular inferior con -inf en posiciones futuras.
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

/// Scaled Dot-Product Attention:
/// Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) + mask ) @ V
pub fn scaled_dot_product_attention(
    q: &Tensor, // [Batch, Heads, Seq_Q, Dim]
    k: &Tensor, // [Batch, Heads, Seq_K, Dim]
    v: &Tensor, // [Batch, Heads, Seq_K, Dim]
    is_causal: bool,
) -> Result<Tensor> {
    let (_, _, seq_len_q, head_dim) = q.dims4()?;
    let (_, _, seq_len_k, _) = k.dims4()?;

    // 1. Q @ K^T -> [Batch, Heads, Seq_Q, Seq_K]
    let k_t = k.transpose(2, 3)?;
    let scale = (head_dim as f64).sqrt();
    let scores = (q.matmul(&k_t)? / scale)?;

    // 2. Aplicar máscara causal si se solicita y Seq_Q > 1 (modo Prefill)
    let scores = if is_causal && seq_len_q > 1 {
        let mask = create_causal_mask(seq_len_q, seq_len_k, q.device())?;
        scores.broadcast_add(&mask)?
    } else {
        scores
    };

    // 3. Softmax a lo largo de la dimensión de las llaves (dim 3)
    let weights = candle_nn::ops::softmax(&scores, 3)?;

    // 4. Weights @ V -> [Batch, Heads, Seq_Q, Dim]
    weights.matmul(v)
}

/// Estructura para el Key-Value Cache (KV-Cache)
/// En inferencia de LLMs, almacena las proyecciones K y V de todos los tokens anteriores.
#[derive(Debug, Default)]
pub struct KvCache {
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Añade los nuevos tensores K y V a lo largo de la dimensión de secuencia (dim 2).
    /// Retorna los tensores concatenados completos (historial + nuevo token).
    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> Result<(Tensor, Tensor)> {
        match (&self.k, &self.v) {
            (None, None) => {
                self.k = Some(new_k.clone());
                self.v = Some(new_v.clone());
                Ok((new_k.clone(), new_v.clone()))
            }
            (Some(past_k), Some(past_v)) => {
                // Dimensión 2 = Secuencia: [Batch, Heads, Seq_len, Dim]
                let full_k = Tensor::cat(&[past_k, new_k], 2)?;
                let full_v = Tensor::cat(&[past_v, new_v], 2)?;
                self.k = Some(full_k.clone());
                self.v = Some(full_v.clone());
                Ok((full_k, full_v))
            }
            _ => candle_core::bail!("Inconsistencia en el estado del KV-Cache"),
        }
    }

    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(t) => t.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 05: ARQUITECTURA TRANSFORMER Y MEMORIA KV-CACHE EN RUST");
    println!("==============================================================================");

    let dev = Device::Cpu;

    // -------------------------------------------------------------------------
    // 1. Fundamento de Scaled Dot-Product Attention y Máscara Causal
    // -------------------------------------------------------------------------
    println!("\n[1] Scaled Dot-Product Attention con Máscara Causal (Prefill):");
    let batch_size = 1;
    let n_heads = 2;
    let seq_len = 3;
    let head_dim = 4;

    // Tensores simulados de Q, K, V
    let q = Tensor::ones((batch_size, n_heads, seq_len, head_dim), candle_core::DType::F32, &dev)?;
    let k = Tensor::ones((batch_size, n_heads, seq_len, head_dim), candle_core::DType::F32, &dev)?;
    let v = Tensor::ones((batch_size, n_heads, seq_len, head_dim), candle_core::DType::F32, &dev)?;

    let out = scaled_dot_product_attention(&q, &k, &v, true)?;
    println!("   Shape de salida Attention: {:?}", out.shape());
    println!("   Atención calculada exitosamente respetando causalidad temporal.");

    // -------------------------------------------------------------------------
    // 2. Rotary Position Embeddings (RoPE)
    // -------------------------------------------------------------------------
    println!("\n[2] Rotary Position Embeddings (RoPE):");
    let pos_0 = apply_rope(&q, 0)?;
    let pos_10 = apply_rope(&q, 10)?;
    println!("   Q rotado en posición 0  (primeros 4 valores): {:?}", 
        &pos_0.flatten_all()?.to_vec1::<f32>()?[0..4]);
    println!("   Q rotado en posición 10 (primeros 4 valores): {:?}", 
        &pos_10.flatten_all()?.to_vec1::<f32>()?[0..4]);
    println!("   -> Los vectores se desfasan según su posición angular relativa.");

    // -------------------------------------------------------------------------
    // 3. Simulación del KV-Cache: Prefill vs Decode autorregresivo
    // -------------------------------------------------------------------------
    println!("\n[3] Inferencia Autorregresiva con KV-Cache:");
    println!("   Supongamos que el usuario envía un prompt de 4 tokens.");
    println!("   Fase 1: PREFILL (procesar los 4 tokens juntos y llenar el KV-Cache)");

    let mut cache = KvCache::new();

    let prompt_len = 4;
    let prompt_k = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, prompt_len, head_dim), &dev)?;
    let prompt_v = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, prompt_len, head_dim), &dev)?;
    let prompt_q = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, prompt_len, head_dim), &dev)?;

    let (cached_k, cached_v) = cache.append(&prompt_k, &prompt_v)?;
    let out_prefill = scaled_dot_product_attention(&prompt_q, &cached_k, &cached_v, true)?;
    println!("   Prefill completado. Salida: {:?}. Longitud en Cache: {}", out_prefill.shape(), cache.seq_len());

    println!("\n   Fase 2: DECODE (generando 3 tokens nuevos, uno por uno)");
    for step in 1..=3 {
        // En cada paso de generación, la entrada es de longitud 1 (un solo token!)
        let new_q = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, 1, head_dim), &dev)?;
        let new_k = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, 1, head_dim), &dev)?;
        let new_v = Tensor::randn(0.0f32, 1.0f32, (1, n_heads, 1, head_dim), &dev)?;

        // Rota con el offset correcto (prompt_len + step - 1)
        let current_pos = prompt_len + step - 1;
        let new_q_rope = apply_rope(&new_q, current_pos)?;
        let new_k_rope = apply_rope(&new_k, current_pos)?;

        // Añadir K y V al cache acumulado
        let (full_k, full_v) = cache.append(&new_k_rope, &new_v)?;

        // Notar: is_causal = false porque new_q tiene Seq_Q = 1 y puede atender
        // a TODOS los tokens anteriores almacenados en full_k!
        let out_step = scaled_dot_product_attention(&new_q_rope, &full_k, &full_v, false)?;

        println!("   Paso {}: Token generado. Q shape: {:?}, Cache K shape: {:?}, Salida: {:?}",
            step, new_q_rope.shape(), full_k.shape(), out_step.shape());
    }

    println!("\n[4] Análisis de Complejidad:");
    println!("   - Sin KV-Cache: Cada token requeriría recomputar Q, K, V para todos los N tokens previos -> O(N^2) flops.");
    println!("   - Con KV-Cache: Solo computamos Q, K, V para el nuevo token -> O(1) proyecciones y O(N) atención.");
    println!("   - VRAM consumida por el Cache: 2 * (Capas * Cabezas * Seq_len * Head_dim * Bytes_por_float).");
    println!("==============================================================================");
    Ok(())
}
