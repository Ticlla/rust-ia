//! Configuración del Enterprise Hybrid RAG Engine

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    /// Dimensión del vector denso de embeddings (ej. 64, 128, 384)
    pub embedding_dim: usize,
    /// Tamaño máximo de cada chunk en caracteres
    pub chunk_size: usize,
    /// Solapamiento (overlap) entre chunks consecutivos
    pub chunk_overlap: usize,
    /// Parámetro k1 de BM25 (saturación de frecuencia de término)
    pub bm25_k1: f32,
    /// Parámetro b de BM25 (penalización por longitud de documento)
    pub bm25_b: f32,
    /// Constante de suavizado para Reciprocal Rank Fusion (típicamente 60.0)
    pub rrf_k: f32,
    /// Peso relativo de la búsqueda densa en RRF (0.0 a 1.0)
    pub dense_weight: f32,
    /// Peso relativo de la búsqueda léxica BM25 en RRF (0.0 a 1.0)
    pub sparse_weight: f32,
    /// Puerto del servidor HTTP
    pub server_port: u16,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            chunk_size: 250,
            chunk_overlap: 50,
            bm25_k1: 1.5,
            bm25_b: 0.75,
            rrf_k: 60.0,
            dense_weight: 0.6,
            sparse_weight: 0.4,
            server_port: 3000,
        }
    }
}
