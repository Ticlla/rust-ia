//! Enrutador de la API Axum.

use super::handlers::{AppState, events_stream_handler, health_handler, run_workflow_handler};
use axum::{
    Router,
    routing::{get, post},
};
use std::sync::Arc;

/// Construye las rutas de la aplicación web del orquestador.
pub fn build_router(app_state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/orchestrator/run", post(run_workflow_handler))
        .route("/v1/orchestrator/events", get(events_stream_handler))
        .with_state(app_state)
}
