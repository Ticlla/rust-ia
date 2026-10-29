//! Configuración del LLM Serving Gateway

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatewayConfig {
    pub host: String,
    pub port: u16,
    pub model_id: String,
    /// Dimensión de embedding interno del modelo
    pub embed_dim: usize,
    /// Número de cabezas de atención
    pub n_heads: usize,
    /// Número de capas Transformer
    pub n_layers: usize,
    /// Tamaño de vocabulario
    pub vocab_size: usize,
    /// Tamaño de cada bloque físico de KV-Cache en tokens (ej. 16 tokens por bloque)
    pub block_size: usize,
    /// Número total de bloques físicos asignables en memoria
    pub total_blocks: usize,
    /// Límite máximo de peticiones concurrentes en batch continuo
    pub max_batch_size: usize,
    /// Token ID de parada (EOS)
    pub eos_token_id: u32,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: std::env::var("PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8081),
            model_id: "smollm2-135m".to_string(),
            embed_dim: 32,
            n_heads: 4,
            n_layers: 2,
            vocab_size: 50,
            block_size: 8,     // 8 tokens por bloque físico
            total_blocks: 128, // 128 bloques * 8 = 1024 tokens de capacidad en VRAM/RAM
            max_batch_size: 16,
            eos_token_id: 2,
        }
    }
}
