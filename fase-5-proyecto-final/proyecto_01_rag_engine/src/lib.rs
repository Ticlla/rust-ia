//! Enterprise Hybrid RAG Engine
//! Motor de Búsqueda Semántica Híbrida de Nivel Empresarial en Rust

pub mod api;
pub mod config;
pub mod embeddings;
pub mod engine;
pub mod ingestion;
pub mod reranking;
pub mod search;

pub use config::RagConfig;
pub use engine::{EnterpriseRagEngine, SearchHit, SearchMode};
pub use ingestion::{Document, DocumentChunk, TextChunker};
