//! Ensamblado de rutas Axum con middlewares de CORS y rastreo.

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;

use crate::api::handlers::{
    classify_vision_handler, health_handler, infer_audio_handler, metrics_handler, AppState,
};

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/v1/vision/classify", post(classify_vision_handler))
        .route("/v1/audio/infer", post(infer_audio_handler))
        .route("/health", get(health_handler))
        .route("/v1/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/metrics", get(metrics_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
