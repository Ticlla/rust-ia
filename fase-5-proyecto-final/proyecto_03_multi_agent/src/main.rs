//! Servidor de Inferencia y Orquestación Multi-Agente en Axum.

use proyecto_03_multi_agent::api::{AppState, build_router};
use proyecto_03_multi_agent::create_default_system;
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Inicializando Autonomous Agent & Multi-Agent Orchestration Framework...");

    let (orchestrator, event_bus) = create_default_system();

    let app_state = Arc::new(AppState {
        orchestrator,
        event_bus,
    });

    let router = build_router(app_state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8082);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("🚀 Orquestador Multi-Agente escuchando en http://{}", addr);
    info!("📡 Endpoints disponibles:");
    info!("  - Health Check:   GET  http://{}/health", addr);
    info!(
        "  - Run Workflow:   POST http://{}/v1/orchestrator/run",
        addr
    );
    info!(
        "  - Event Streaming: GET http://{}/v1/orchestrator/events (SSE)",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
