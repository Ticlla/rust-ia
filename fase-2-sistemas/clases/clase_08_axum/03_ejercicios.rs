//! ============================================================================
//! 🎯 FASE 2 - CLASE 08: Retos Prácticos con Servidores Axum en IA
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c08_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c08_ejercicios

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;

// =============================================================================
// ESTADO Y DTOs
// =============================================================================

#[derive(Clone)]
pub struct ServidorState {
    pub pesos: Arc<RwLock<Vec<f32>>>,
    pub contador: Arc<AtomicUsize>,
}

#[derive(Deserialize)]
pub struct EmbedRequest {
    pub texto: String,
}

#[derive(Serialize, PartialEq, Debug)]
pub struct EmbedResponse {
    pub vector: Vec<f32>,
    pub dimension: usize,
}

#[derive(Deserialize)]
pub struct ActualizarPesosRequest {
    pub nuevos_pesos: Vec<f32>,
}

#[derive(Serialize)]
pub struct StatsResponse {
    pub total_consultas: usize,
    pub dimension_modelo: usize,
}

// =============================================================================
// HANDLERS
// =============================================================================

/// Genera un embedding semántico para el texto recibido y suma al contador atómico.
pub async fn embed_handler(
    State(state): State<ServidorState>,
    Json(payload): Json<EmbedRequest>,
) -> (StatusCode, Json<EmbedResponse>) {
    state.contador.fetch_add(1, Ordering::Relaxed);

    let len = payload.texto.len() as f32;
    let pesos = state.pesos.read().unwrap();

    let vector: Vec<f32> = pesos.iter().map(|&w| (w * len) % 1.0).collect();
    let dim = vector.len();

    (StatusCode::OK, Json(EmbedResponse { vector, dimension: dim }))
}

/// Actualiza los pesos del modelo en caliente usando exclusión mutua de escritura.
pub async fn actualizar_pesos_handler(
    State(state): State<ServidorState>,
    Json(payload): Json<ActualizarPesosRequest>,
) -> StatusCode {
    let mut guard = state.pesos.write().unwrap();
    *guard = payload.nuevos_pesos;
    StatusCode::ACCEPTED
}

/// Retorna estadísticas de uso del microservicio.
pub async fn stats_handler(State(state): State<ServidorState>) -> Json<StatsResponse> {
    let total = state.contador.load(Ordering::Relaxed);
    let dim = state.pesos.read().unwrap().len();

    Json(StatsResponse {
        total_consultas: total,
        dimension_modelo: dim,
    })
}

/// Construye la aplicación Axum configurada con sus rutas y estado.
pub fn crear_app(state: ServidorState) -> Router {
    Router::new()
        .route("/embed", post(embed_handler))
        .route("/actualizar", post(actualizar_pesos_handler))
        .route("/stats", get(stats_handler))
        .with_state(state)
}

// =============================================================================
// MAIN DE DEMOSTRACIÓN
// =============================================================================

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Microservicio de Embeddings con Axum");
    println!("============================================================\n");

    let state = ServidorState {
        pesos: Arc::new(RwLock::new(vec![0.1, 0.5, 0.9])),
        contador: Arc::new(AtomicUsize::new(0)),
    };

    let app = crear_app(state.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    println!("Servidor listo en: http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    println!("\nProbando endpoints directamente con llamadas a los handlers...");

    // Test directo 1: Generar embedding
    let (_, res1) = embed_handler(
        State(state.clone()),
        Json(EmbedRequest { texto: "Rust y LLMs".to_string() }),
    )
    .await;
    println!("Respuesta /embed: {:?}", res1.0);

    // Test directo 2: Actualizar pesos en caliente
    let status = actualizar_pesos_handler(
        State(state.clone()),
        Json(ActualizarPesosRequest { nuevos_pesos: vec![0.3, 0.7] }),
    )
    .await;
    println!("Respuesta /actualizar: Status {:?}", status);

    // Test directo 3: Consultar stats
    let stats = stats_handler(State(state.clone())).await;
    println!("Respuesta /stats: Total consultas = {}, Nueva Dim = {}", stats.total_consultas, stats.dimension_modelo);

    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_embed_genera_dimension_correcta() {
        let state = ServidorState {
            pesos: Arc::new(RwLock::new(vec![0.1, 0.2, 0.3])),
            contador: Arc::new(AtomicUsize::new(0)),
        };

        let (status, Json(res)) = embed_handler(
            State(state.clone()),
            Json(EmbedRequest { texto: "hola".to_string() }),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(res.dimension, 3);
        assert_eq!(state.contador.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn test_actualizar_pesos_en_caliente() {
        let state = ServidorState {
            pesos: Arc::new(RwLock::new(vec![1.0])),
            contador: Arc::new(AtomicUsize::new(0)),
        };

        let status = actualizar_pesos_handler(
            State(state.clone()),
            Json(ActualizarPesosRequest { nuevos_pesos: vec![2.0, 3.0] }),
        )
        .await;

        assert_eq!(status, StatusCode::ACCEPTED);
        assert_eq!(state.pesos.read().unwrap().len(), 2);
    }
}
