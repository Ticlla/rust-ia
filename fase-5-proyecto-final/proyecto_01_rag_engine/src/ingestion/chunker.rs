//! Segmentador de Texto Inteligente con Overlap

use super::document::{Document, DocumentChunk};

pub struct TextChunker {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl TextChunker {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        assert!(
            chunk_size > chunk_overlap,
            "chunk_size debe ser mayor que chunk_overlap"
        );
        Self {
            chunk_size,
            chunk_overlap,
        }
    }

    /// Divide un documento en fragmentos solapados respetando límites de oraciones o espacios
    pub fn chunk_document(&self, doc: &Document) -> Vec<DocumentChunk> {
        let content = doc.content.trim();
        if content.is_empty() {
            return Vec::new();
        }

        // Si el contenido completo entra en un solo chunk
        if content.len() <= self.chunk_size {
            return vec![DocumentChunk {
                id: format!("{}_chunk_0", doc.id),
                doc_id: doc.id.clone(),
                doc_title: doc.title.clone(),
                chunk_index: 0,
                text: content.to_string(),
                metadata: doc.metadata.clone(),
            }];
        }

        let mut chunks = Vec::new();
        let step = self.chunk_size - self.chunk_overlap;
        let mut start_idx = 0;
        let mut chunk_index = 0;

        while start_idx < content.len() {
            let mut end_idx = (start_idx + self.chunk_size).min(content.len());

            // Si no estamos al final del texto, buscar un límite natural (espacio o punto)
            if end_idx < content.len() {
                let slice = &content[start_idx..end_idx];
                if let Some(pos) = slice.rfind(|c: char| c == '.' || c == '\n' || c == ' ') {
                    if pos > self.chunk_overlap {
                        end_idx = start_idx + pos + 1;
                    }
                }
            }

            let chunk_text = content[start_idx..end_idx].trim();
            if !chunk_text.is_empty() {
                chunks.push(DocumentChunk {
                    id: format!("{}_chunk_{}", doc.id, chunk_index),
                    doc_id: doc.id.clone(),
                    doc_title: doc.title.clone(),
                    chunk_index,
                    text: chunk_text.to_string(),
                    metadata: doc.metadata.clone(),
                });
                chunk_index += 1;
            }

            if end_idx >= content.len() {
                break;
            }

            start_idx += step;
        }

        chunks
    }
}
