//! Manejadores de Rutas HTTP de Axum

use super::models::{HealthResponse, IngestRequest, IngestResponse, SearchRequest, SearchResponse};
use crate::engine::EnterpriseRagEngine;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;

pub struct AppState {
    pub engine: Arc<RwLock<EnterpriseRagEngine>>,
    pub start_time: Instant,
    pub searches_total: AtomicU64,
    pub ingests_total: AtomicU64,
}

impl AppState {
    pub fn new(engine: Arc<RwLock<EnterpriseRagEngine>>) -> Self {
        Self {
            engine,
            start_time: Instant::now(),
            searches_total: AtomicU64::new(0),
            ingests_total: AtomicU64::new(0),
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
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

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let engine = state.engine.read().await;
    let stats = engine.stats();
    let uptime_secs = state.start_time.elapsed().as_secs();
    let searches = state.searches_total.load(Ordering::Relaxed);
    let ingests = state.ingests_total.load(Ordering::Relaxed);

    let body = format!(
        "# HELP rag_engine_uptime_seconds Tiempo de actividad del motor RAG en segundos\n\
         # TYPE rag_engine_uptime_seconds gauge\n\
         rag_engine_uptime_seconds {}\n\n\
         # HELP rag_engine_documents_total Total de documentos cargados en el índice RAG\n\
         # TYPE rag_engine_documents_total gauge\n\
         rag_engine_documents_total {}\n\n\
         # HELP rag_engine_chunks_total Total de fragmentos de texto vectorizados en memoria\n\
         # TYPE rag_engine_chunks_total gauge\n\
         rag_engine_chunks_total {}\n\n\
         # HELP rag_engine_dense_vectors_total Total de vectores densos indexados\n\
         # TYPE rag_engine_dense_vectors_total gauge\n\
         rag_engine_dense_vectors_total {}\n\n\
         # HELP rag_engine_bm25_terms_total Total de términos indexados en el índice invertido BM25\n\
         # TYPE rag_engine_bm25_terms_total gauge\n\
         rag_engine_bm25_terms_total {}\n\n\
         # HELP rag_engine_searches_total Cantidad acumulada de búsquedas híbridas ejecutadas\n\
         # TYPE rag_engine_searches_total counter\n\
         rag_engine_searches_total {}\n\n\
         # HELP rag_engine_ingests_total Cantidad acumulada de operaciones de ingesta de documentos\n\
         # TYPE rag_engine_ingests_total counter\n\
         rag_engine_ingests_total {}\n",
        uptime_secs,
        stats.total_documents,
        stats.total_chunks,
        stats.dense_vectors_indexed,
        stats.bm25_terms_indexed,
        searches,
        ingests,
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
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
            state.ingests_total.fetch_add(1, Ordering::Relaxed);
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
            state.searches_total.fetch_add(1, Ordering::Relaxed);
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
