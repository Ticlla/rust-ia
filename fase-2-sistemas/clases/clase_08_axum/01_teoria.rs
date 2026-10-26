//! ============================================================================
//! 🦀 FASE 2 - CLASE 08: Servidores de Inferencia de IA con `Axum`
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Qué es `Axum`: el framework web oficial del equipo de Tokio.
//! 2. Extractor de estado compartido: `State(Arc<AppState>)` con `RwLock`.
//! 3. Extractor de payloads JSON: `Json(PeticionPredict)`.
//! 4. Servir predicciones de IA con ultra baja latencia y alta concurrencia.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c08_teoria

use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

/// Estructura que representa los pesos del modelo de IA en memoria RAM.
#[derive(Debug, Clone)]
struct ModeloIA {
    version: usize,
    pesos: Vec<f32>,
}

impl ModeloIA {
    fn predecir(&self, entrada: &[f32]) -> f32 {
        self.pesos.iter().zip(entrada.iter()).map(|(w, x)| w * x).sum()
    }
}

/// Estado global de la aplicación web compartido entre todas las peticiones HTTP.
struct AppState {
    modelo: Arc<RwLock<ModeloIA>>,
    nombre_servicio: String,
}

// =============================================================================
// DTOs (Data Transfer Objects) SERIALIZABLES CON SERDE
// =============================================================================

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    servicio: String,
    modelo_version: usize,
}

#[derive(Serialize, Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediccion: f32,
    modelo_version: usize,
}

// =============================================================================
// HANDLERS ASÍNCRONOS DE AXUM
// =============================================================================

/// Endpoint GET /health: Retorna el estado de salud del servidor y la versión del modelo.
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let version = {
        let m = state.modelo.read().unwrap();
        m.version
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        servicio: state.nombre_servicio.clone(),
        modelo_version: version,
    })
}

/// Endpoint POST /predict: Recibe un vector de características y devuelve la predicción.
async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> Json<PredictResponse> {
    // 💡 Leemos concurrentemente los pesos con RwLock (cero bloqueos entre usuarios):
    let (pred, version) = {
        let m = state.modelo.read().unwrap();
        let resultado = m.predecir(&payload.features);
        (resultado, m.version)
    };

    Json(PredictResponse {
        prediccion: pred,
        modelo_version: version,
    })
}

// =============================================================================
// FUNCIÓN PRINCIPAL
// =============================================================================

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🚀 FASE 2 - CLASE 08: Servidor Web de Inferencia con Axum");
    println!("============================================================\n");

    // 1. Inicializamos el modelo en memoria compartida (Clases 03 y 04):
    let estado_compartido = Arc::new(AppState {
        modelo: Arc::new(RwLock::new(ModeloIA {
            version: 1,
            pesos: vec![0.5, 2.0, -1.0], // y = 0.5*x0 + 2.0*x1 - 1.0*x2
        })),
        nombre_servicio: "Microservicio-Inferencia-Rust".to_string(),
    });

    // 2. Construimos el enrutador de Axum asociando rutas con handlers y el estado:
    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .with_state(estado_compartido.clone());

    // 3. Abrimos el puerto TCP dinámico en localhost:
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    println!("🌐 Servidor Axum iniciado y escuchando en: http://{}", addr);

    // 4. Lanzamos el servidor en segundo plano con tokio::spawn:
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // =========================================================================
    // SIMULACIÓN: ENVIAR PETICIONES HTTP REALES POR EL SOCKET TCP
    // =========================================================================
    println!("\n--- 1. Enviando Petición GET /health ---");
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let get_req = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(get_req.as_bytes()).await.unwrap();

    let mut response_buf = Vec::new();
    stream.read_to_end(&mut response_buf).await.unwrap();
    let response_str = String::from_utf8_lossy(&response_buf);
    println!("Respuesta HTTP recibida:\n{}", response_str);

    println!("--- 2. Enviando Petición POST /predict con JSON [10.0, 5.0, 2.0] ---");
    let body = serde_json::to_string(&PredictRequest {
        features: vec![10.0, 5.0, 2.0],
    })
    .unwrap();

    let post_req = format!(
        "POST /predict HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );

    let mut stream2 = TcpStream::connect(addr).await.unwrap();
    stream2.write_all(post_req.as_bytes()).await.unwrap();

    let mut response_buf2 = Vec::new();
    stream2.read_to_end(&mut response_buf2).await.unwrap();
    let response_str2 = String::from_utf8_lossy(&response_buf2);
    println!("Respuesta HTTP recibida:\n{}", response_str2);
    println!("Esperado: 10*0.5 + 5*2.0 - 2*1.0 = 5.0 + 10.0 - 2.0 = 13.0");

    println!("\n✅ Servidor de Inferencia probado exitosamente.");
}
