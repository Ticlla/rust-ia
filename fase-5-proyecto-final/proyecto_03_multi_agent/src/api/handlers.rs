//! Manejadores HTTP para Axum con soporte de Server-Sent Events (SSE).

use super::models::{HealthResponse, WorkflowRequest, WorkflowResponse};
use crate::events::EventBus;
use crate::orchestrator::SupervisorOrchestrator;
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::IntoResponse,
    Json,
};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::sync::Arc;
use uuid::Uuid;

/// Estado global compartido de la API.
#[derive(Clone)]
pub struct AppState {
    pub orchestrator: Arc<SupervisorOrchestrator>,
    pub event_bus: EventBus,
}

/// Endpoint `GET /health` para monitoreo de salud del servicio.
pub async fn health_handler(State(_state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".into(),
        service: "autonomous-multi-agent-orchestrator".into(),
        active_actors: 3,
        version: env!("CARGO_PKG_VERSION").into(),
    })
}

/// Endpoint `POST /v1/orchestrator/run` para lanzar una meta en el flujo multi-agente.
pub async fn run_workflow_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<WorkflowRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let workflow_id = payload
        .workflow_id
        .unwrap_or_else(|| format!("wf-{}", Uuid::new_v4().simple()));

    let report = state
        .orchestrator
        .run_workflow(&workflow_id, &payload.goal)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(WorkflowResponse {
        workflow_id,
        status: "completed".into(),
        report,
    }))
}

/// Endpoint `GET /v1/orchestrator/events` para streaming reactivo SSE de eventos de agentes.
pub async fn events_stream_handler(
    State(state): State<Arc<AppState>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let mut receiver = state.event_bus.subscribe();

    let stream = async_stream::stream! {
        while let Ok(event) = receiver.recv().await {
            if let Ok(json_str) = serde_json::to_string(&event) {
                yield Ok(Event::default().data(json_str));
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
