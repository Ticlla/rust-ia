//! Orquestador Principal del Enterprise Hybrid RAG Engine

use crate::config::RagConfig;
use crate::embeddings::DenseEmbedder;
use crate::ingestion::{Document, DocumentChunk, TextChunker};
use crate::reranking::CrossEncoderReranker;
use crate::search::{fuse_reciprocal_rank, Bm25Index, DenseIndex};
use candle_core::{Device, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    Dense,
    Sparse,
    Hybrid,
    HybridWithRerank,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub doc_id: String,
    pub doc_title: String,
    pub text: String,
    pub final_score: f32,
    pub rrf_score: Option<f32>,
    pub dense_score: Option<f32>,
    pub bm25_score: Option<f32>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStats {
    pub total_documents: usize,
    pub total_chunks: usize,
    pub embedding_dim: usize,
    pub dense_vectors_indexed: usize,
    pub bm25_terms_indexed: usize,
}

pub struct EnterpriseRagEngine {
    pub config: RagConfig,
    pub chunker: TextChunker,
    pub embedder: DenseEmbedder,
    pub dense_index: DenseIndex,
    pub bm25_index: Bm25Index,
    pub reranker: CrossEncoderReranker,
    pub chunks_store: HashMap<String, DocumentChunk>,
    pub documents_count: usize,
}

impl EnterpriseRagEngine {
    pub fn new(config: RagConfig, dev: &Device) -> Result<Self> {
        let chunker = TextChunker::new(config.chunk_size, config.chunk_overlap);
        let embedder = DenseEmbedder::new(config.embedding_dim, dev)?;
        let dense_index = DenseIndex::new(config.embedding_dim);
        let bm25_index = Bm25Index::new(config.bm25_k1, config.bm25_b);
        let reranker = CrossEncoderReranker::new(dev)?;

        Ok(Self {
            config,
            chunker,
            embedder,
            dense_index,
            bm25_index,
            reranker,
            chunks_store: HashMap::new(),
            documents_count: 0,
        })
    }

    /// Ingesta una lista de documentos dividiéndolos en fragmentos e indexándolos en ambos motores
    pub fn ingest_documents(&mut self, docs: Vec<Document>) -> Result<usize> {
        let mut new_chunks_count = 0;

        for doc in docs {
            let chunks = self.chunker.chunk_document(&doc);
            for chunk in chunks {
                // 1. Generar embedding denso con Candle
                let embedding = self.embedder.embed(&chunk.text)?;

                // 2. Indexar en índice denso SIMD
                self.dense_index.insert(chunk.id.clone(), embedding);

                // 3. Indexar en índice léxico BM25
                self.bm25_index.insert(chunk.id.clone(), &chunk.text);

                // 4. Almacenar el chunk original en memoria
                self.chunks_store.insert(chunk.id.clone(), chunk);
                new_chunks_count += 1;
            }
            self.documents_count += 1;
        }

        Ok(new_chunks_count)
    }

    /// Búsqueda unificada multimodelo
    pub fn search(&self, query: &str, top_k: usize, mode: SearchMode) -> Result<Vec<SearchHit>> {
        let fetch_k = (top_k * 3).max(10); // Sobre-muestreo inicial para fusión y reranking

        match mode {
            SearchMode::Dense => {
                let query_embedding = self.embedder.embed(query)?;
                let dense_results = self.dense_index.search(&query_embedding, top_k);

                Ok(dense_results
                    .into_iter()
                    .filter_map(|(id, score)| {
                        self.chunks_store.get(&id).map(|c| SearchHit {
                            chunk_id: c.id.clone(),
                            doc_id: c.doc_id.clone(),
                            doc_title: c.doc_title.clone(),
                            text: c.text.clone(),
                            final_score: score,
                            rrf_score: None,
                            dense_score: Some(score),
                            bm25_score: None,
                            metadata: c.metadata.clone(),
                        })
                    })
                    .collect())
            }

            SearchMode::Sparse => {
                let bm25_results = self.bm25_index.search(query, top_k);

                Ok(bm25_results
                    .into_iter()
                    .filter_map(|(id, score)| {
                        self.chunks_store.get(&id).map(|c| SearchHit {
                            chunk_id: c.id.clone(),
                            doc_id: c.doc_id.clone(),
                            doc_title: c.doc_title.clone(),
                            text: c.text.clone(),
                            final_score: score,
                            rrf_score: None,
                            dense_score: None,
                            bm25_score: Some(score),
                            metadata: c.metadata.clone(),
                        })
                    })
                    .collect())
            }

            SearchMode::Hybrid => {
                let query_embedding = self.embedder.embed(query)?;
                let dense_results = self.dense_index.search(&query_embedding, fetch_k);
                let sparse_results = self.bm25_index.search(query, fetch_k);

                let fused = fuse_reciprocal_rank(
                    &dense_results,
                    &sparse_results,
                    self.config.rrf_k,
                    self.config.dense_weight,
                    self.config.sparse_weight,
                    top_k,
                );

                Ok(fused
                    .into_iter()
                    .filter_map(|r| {
                        self.chunks_store.get(&r.chunk_id).map(|c| SearchHit {
                            chunk_id: c.id.clone(),
                            doc_id: c.doc_id.clone(),
                            doc_title: c.doc_title.clone(),
                            text: c.text.clone(),
                            final_score: r.rrf_score,
                            rrf_score: Some(r.rrf_score),
                            dense_score: r.dense_score,
                            bm25_score: r.bm25_score,
                            metadata: c.metadata.clone(),
                        })
                    })
                    .collect())
            }

            SearchMode::HybridWithRerank => {
                let query_embedding = self.embedder.embed(query)?;
                let dense_results = self.dense_index.search(&query_embedding, fetch_k);
                let sparse_results = self.bm25_index.search(query, fetch_k);

                let fused = fuse_reciprocal_rank(
                    &dense_results,
                    &sparse_results,
                    self.config.rrf_k,
                    self.config.dense_weight,
                    self.config.sparse_weight,
                    fetch_k,
                );

                let candidate_triplets: Vec<(String, String, f32)> = fused
                    .iter()
                    .filter_map(|r| {
                        self.chunks_store
                            .get(&r.chunk_id)
                            .map(|c| (c.id.clone(), c.text.clone(), r.rrf_score))
                    })
                    .collect();

                let reranked = self.reranker.rerank(query, &candidate_triplets, top_k)?;

                Ok(reranked
                    .into_iter()
                    .filter_map(|cand| {
                        self.chunks_store.get(&cand.chunk_id).map(|c| SearchHit {
                            chunk_id: c.id.clone(),
                            doc_id: c.doc_id.clone(),
                            doc_title: c.doc_title.clone(),
                            text: c.text.clone(),
                            final_score: cand.relevance_score,
                            rrf_score: candidate_triplets
                                .iter()
                                .find(|(id, _, _)| id == &cand.chunk_id)
                                .map(|(_, _, s)| *s),
                            dense_score: None,
                            bm25_score: None,
                            metadata: c.metadata.clone(),
                        })
                    })
                    .collect())
            }
        }
    }

    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_documents: self.documents_count,
            total_chunks: self.chunks_store.len(),
            embedding_dim: self.config.embedding_dim,
            dense_vectors_indexed: self.dense_index.len(),
            bm25_terms_indexed: self.bm25_index.inverted_index.len(),
        }
    }
}
