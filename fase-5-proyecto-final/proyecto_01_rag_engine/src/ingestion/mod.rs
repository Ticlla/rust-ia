//! Módulo de Ingesta y Procesamiento de Documentos

pub mod chunker;
pub mod document;

pub use chunker::TextChunker;
pub use document::{Document, DocumentChunk};
