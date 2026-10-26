//! Extractor de Embeddings Densos L2-Normalizados usando Candle

use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Linear, Module};

pub struct DenseEmbedder {
    pub embedding: Embedding,
    pub projection: Linear,
    pub dim: usize,
    pub dev: Device,
}

impl DenseEmbedder {
    pub fn new(dim: usize, dev: &Device) -> Result<Self> {
        let vocab_size = 5000;
        // Inicialización determinista y reproducible de pesos
        let mut emb_data = Vec::with_capacity(vocab_size * dim);
        for i in 0..(vocab_size * dim) {
            let val = (i as f32 * 0.1234).sin() * 0.5;
            emb_data.push(val);
        }
        let w_emb = Tensor::from_vec(emb_data, (vocab_size, dim), dev)?;
        let embedding = Embedding::new(w_emb, dim);

        let mut proj_data = Vec::with_capacity(dim * dim);
        for i in 0..(dim * dim) {
            let val = if i % (dim + 1) == 0 {
                1.0f32
            } else {
                (i as f32 * 0.5678).cos() * 0.1
            };
            proj_data.push(val);
        }
        let w_proj = Tensor::from_vec(proj_data, (dim, dim), dev)?;
        let projection = Linear::new(w_proj, None);

        Ok(Self {
            embedding,
            projection,
            dim,
            dev: dev.clone(),
        })
    }

    /// Convierte un texto en una secuencia de token IDs mediante hash modular determinista
    fn tokenize(&self, text: &str) -> Vec<u32> {
        let cleaned: String = text
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();

        let words: Vec<&str> = cleaned.split_whitespace().collect();
        if words.is_empty() {
            return vec![0];
        }

        words
            .into_iter()
            .map(|w| {
                // Algoritmo FNV-1a simple para hashing determinista de tokens
                let mut hash: u32 = 2166136261;
                for byte in w.bytes() {
                    hash ^= byte as u32;
                    hash = hash.wrapping_mul(16777619);
                }
                (hash % 4999) + 1
            })
            .collect()
    }

    /// Genera un embedding normalizado L2 para una cadena de texto
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let token_ids = self.tokenize(text);
        let _n_tokens = token_ids.len();

        let input_tensor = Tensor::new(&token_ids[..], &self.dev)?; // [n_tokens]
        let token_embeddings = self.embedding.forward(&input_tensor)?; // [n_tokens, dim]

        // Mean Pooling: promedio a lo largo de los tokens
        let pooled = token_embeddings.mean(0)?; // [dim]

        // Proyección neuronal no lineal
        let projected = self.projection.forward(&pooled.unsqueeze(0)?)?.squeeze(0)?;
        let activated = candle_nn::ops::silu(&projected)?;

        // L2 Normalización: v / ||v||
        let sq_sum = activated.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let norm = (sq_sum.sqrt()).max(1e-8);
        let normalized = (activated / (norm as f64))?;

        normalized.to_vec1::<f32>()
    }
}
