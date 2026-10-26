//! Manejadores de Rutas HTTP de Axum

use super::models::{HealthResponse, IngestRequest, IngestResponse, SearchRequest, SearchResponse};
use crate::engine::EnterpriseRagEngine;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

pub struct AppState {
    pub engine: Arc<RwLock<EnterpriseRagEngine>>,
    pub start_time: Instant,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/documents", post(ingest_handler))
        .route("/api/v1/search", post(search_handler))
        .route("/api/v1/stats", get(stats_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.read().await;
    let stats = engine.stats();
    let uptime_secs = state.start_time.elapsed().as_secs();

    Json(HealthResponse {
        status: "healthy".to_string(),
        uptime_secs,
        stats,
    })
}

async fn stats_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.read().await;
    Json(engine.stats())
}

async fn ingest_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<IngestRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let start = Instant::now();
    let doc_count = payload.documents.len();

    let mut engine = state.engine.write().await;
    match engine.ingest_documents(payload.documents) {
        Ok(chunks_created) => {
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok((
                StatusCode::CREATED,
                Json(IngestResponse {
                    status: "success".to_string(),
                    documents_received: doc_count,
                    chunks_created,
                    duration_ms,
                }),
            ))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error durante ingesta: {}", e),
        )),
    }
}

async fn search_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let start = Instant::now();
    let engine = state.engine.read().await;

    match engine.search(&payload.query, payload.top_k, payload.mode) {
        Ok(hits) => {
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            let total_hits = hits.len();
            Ok(Json(SearchResponse {
                query: payload.query,
                total_hits,
                mode: payload.mode,
                duration_ms,
                hits,
            }))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Error durante búsqueda: {}", e),
        )),
    }
}
