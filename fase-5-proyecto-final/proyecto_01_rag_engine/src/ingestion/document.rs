//! Estructuras de Datos de Documentos y Chunks

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Documento fuente original ingresado por el usuario o pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl Document {
    pub fn new(
        id: impl Into<String>,
        title: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Fragmento granular (Chunk) generado tras la partición de un documento
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub doc_id: String,
    pub doc_title: String,
    pub chunk_index: usize,
    pub text: String,
    pub metadata: HashMap<String, String>,
}
