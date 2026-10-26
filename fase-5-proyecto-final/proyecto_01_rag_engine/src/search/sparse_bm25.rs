//! Índice Léxico Invertido con Algoritmo de Ranking BM25

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Bm25Index {
    /// Inverted index: término -> lista de (chunk_id, frecuencia_en_chunk)
    pub inverted_index: HashMap<String, Vec<(String, usize)>>,
    /// Longitud de cada documento (número de palabras)
    pub doc_lengths: HashMap<String, usize>,
    /// Longitud total acumulada de todos los documentos
    pub total_length: usize,
    /// Total de documentos indexados
    pub total_docs: usize,
    pub k1: f32,
    pub b: f32,
}

impl Bm25Index {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            inverted_index: HashMap::new(),
            doc_lengths: HashMap::new(),
            total_length: 0,
            total_docs: 0,
            k1,
            b,
        }
    }

    /// Normaliza y divide una cadena en tokens de búsqueda
    pub fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| w.to_string())
            .collect()
    }

    /// Añade un chunk al índice invertido
    pub fn insert(&mut self, chunk_id: String, text: &str) {
        let tokens = Self::tokenize(text);
        let doc_len = tokens.len();

        let mut term_counts: HashMap<String, usize> = HashMap::new();
        for t in tokens {
            *term_counts.entry(t).or_insert(0) += 1;
        }

        for (term, count) in term_counts {
            self.inverted_index
                .entry(term)
                .or_default()
                .push((chunk_id.clone(), count));
        }

        self.doc_lengths.insert(chunk_id, doc_len);
        self.total_length += doc_len;
        self.total_docs += 1;
    }

    /// Calcula la puntuación BM25 para una consulta dada
    pub fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        if self.total_docs == 0 {
            return Vec::new();
        }

        let query_tokens = Self::tokenize(query);
        if query_tokens.is_empty() {
            return Vec::new();
        }

        let avgdl = (self.total_length as f32) / (self.total_docs as f32);
        let mut scores: HashMap<String, f32> = HashMap::new();

        for term in &query_tokens {
            if let Some(postings) = self.inverted_index.get(term) {
                let n_q = postings.len() as f32;
                // IDF de Robertson-Spärck Jones suavizado
                let idf = ((self.total_docs as f32 - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();

                for (chunk_id, freq) in postings {
                    let doc_len = *self.doc_lengths.get(chunk_id).unwrap_or(&1) as f32;
                    let f = *freq as f32;

                    // Denominador BM25: f + k1 * (1 - b + b * (doc_len / avgdl))
                    let denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / avgdl));
                    let term_score = idf * (f * (self.k1 + 1.0)) / denom;

                    *scores.entry(chunk_id.clone()).or_insert(0.0) += term_score;
                }
            }
        }

        let mut results: Vec<(String, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }
}
