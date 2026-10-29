//! Módulos de Búsqueda Densa, Léxica (BM25) y Fusión Híbrida (RRF)

pub mod dense_simd;
pub mod hybrid_fusion;
pub mod sparse_bm25;

pub use dense_simd::DenseIndex;
pub use hybrid_fusion::{HybridSearchResult, fuse_reciprocal_rank};
pub use sparse_bm25::Bm25Index;
