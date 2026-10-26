//! Algoritmo de Fusión Recíproca de Rangos (Reciprocal Rank Fusion - RRF)

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    pub chunk_id: String,
    pub rrf_score: f32,
    pub dense_score: Option<f32>,
    pub dense_rank: Option<usize>,
    pub bm25_score: Option<f32>,
    pub bm25_rank: Option<usize>,
}

/// Fusiona dos listas ordenadas de candidatos usando Reciprocal Rank Fusion (RRF)
///
/// Score RRF = (w_dense / (k + rank_dense)) + (w_sparse / (k + rank_sparse))
pub fn fuse_reciprocal_rank(
    dense_results: &[(String, f32)],
    sparse_results: &[(String, f32)],
    k: f32,
    dense_weight: f32,
    sparse_weight: f32,
    top_k: usize,
) -> Vec<HybridSearchResult> {
    let mut intermediate: HashMap<String, HybridSearchResult> = HashMap::new();

    // 1. Procesar ranking de búsqueda densa
    for (idx, (chunk_id, score)) in dense_results.iter().enumerate() {
        let rank = idx + 1; // 1-indexed
        let rrf_contrib = dense_weight / (k + rank as f32);

        let entry = intermediate
            .entry(chunk_id.clone())
            .or_insert_with(|| HybridSearchResult {
                chunk_id: chunk_id.clone(),
                rrf_score: 0.0,
                dense_score: None,
                dense_rank: None,
                bm25_score: None,
                bm25_rank: None,
            });

        entry.rrf_score += rrf_contrib;
        entry.dense_score = Some(*score);
        entry.dense_rank = Some(rank);
    }

    // 2. Procesar ranking de búsqueda léxica (BM25)
    for (idx, (chunk_id, score)) in sparse_results.iter().enumerate() {
        let rank = idx + 1; // 1-indexed
        let rrf_contrib = sparse_weight / (k + rank as f32);

        let entry = intermediate
            .entry(chunk_id.clone())
            .or_insert_with(|| HybridSearchResult {
                chunk_id: chunk_id.clone(),
                rrf_score: 0.0,
                dense_score: None,
                dense_rank: None,
                bm25_score: None,
                bm25_rank: None,
            });

        entry.rrf_score += rrf_contrib;
        entry.bm25_score = Some(*score);
        entry.bm25_rank = Some(rank);
    }

    // 3. Ordenar descendentemente por score RRF acumulado
    let mut results: Vec<HybridSearchResult> = intermediate.into_values().collect();
    results.sort_by(|a, b| {
        b.rrf_score
            .partial_cmp(&a.rrf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    results.truncate(top_k);
    results
}
