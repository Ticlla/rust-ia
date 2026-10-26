//! Modelos de Datos para la API REST (DTOs)

use crate::engine::{EngineStats, SearchHit, SearchMode};
use crate::ingestion::Document;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestRequest {
    pub documents: Vec<Document>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestResponse {
    pub status: String,
    pub documents_received: usize,
    pub chunks_created: usize,
    pub duration_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    #[serde(default = "default_search_mode")]
    pub mode: SearchMode,
}

fn default_top_k() -> usize {
    5
}

fn default_search_mode() -> SearchMode {
    SearchMode::HybridWithRerank
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub query: String,
    pub total_hits: usize,
    pub mode: SearchMode,
    pub duration_ms: f64,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime_secs: u64,
    pub stats: EngineStats,
}
