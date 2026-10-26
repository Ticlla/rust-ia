//! ============================================================================
//! 🌐 FASE 2 - CLASE 08: Servidor Web de Inferencia en Vivo
//! ============================================================================
//!
//! Este binario inicia un servidor HTTP real y permanente en:
//!     http://127.0.0.1:3000
//!
//! Se quedará corriendo en tu terminal escuchando peticiones hasta que presiones
//! Ctrl+C para detenerlo.
//!
//! Para ejecutarlo:
//!     cargo run --bin c08_servidor

use axum::{
    extract::State,
    http::StatusCode,
    response::Html,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use tokio::net::TcpListener;

// =============================================================================
// ESTADO COMPARTIDO (MODELO + CONTADORES)
// =============================================================================

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

struct AppState {
    modelo: Arc<RwLock<ModeloIA>>,
    total_consultas: Arc<AtomicUsize>,
}

// =============================================================================
// DTOs (REQUEST / RESPONSE JSON)
// =============================================================================

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    modelo_version: usize,
    pesos_actuales: Vec<f32>,
    total_predicciones_servidas: usize,
}

#[derive(Deserialize)]
struct PredictRequest {
    features: Vec<f32>,
}

#[derive(Serialize)]
struct PredictResponse {
    prediccion: f32,
    dimension_entrada: usize,
    modelo_version: usize,
}

#[derive(Deserialize)]
struct UpdateModelRequest {
    nuevos_pesos: Vec<f32>,
}

// =============================================================================
// HANDLERS HTTP
// =============================================================================

/// Ruta raíz GET /: Página de bienvenida con instrucciones para el usuario.
async fn root_handler() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html lang="es">
    <head><meta charset="utf-8"><title>Servidor de Inferencia Rust</title></head>
    <body style="font-family: sans-serif; max-width: 800px; margin: 40px auto; line-height: 1.6;">
        <h1>🦀 Servidor de Inferencia de IA en Rust (Axum + Tokio)</h1>
        <p>¡El servidor está <strong>100% activo y en vivo</strong> en tu máquina!</p>
        <h2>Rutas disponibles:</h2>
        <ul>
            <li><code>GET <a href="/health">/health</a></code> - Ver estado del modelo y métricas.</li>
            <li><code>POST /predict</code> - Enviar vector JSON para inferencia.</li>
            <li><code>POST /update</code> - Actualizar los pesos del modelo en caliente.</li>
        </ul>
        <hr>
        <p><em>Prueba enviar peticiones desde tu terminal con curl.</em></p>
    </body>
    </html>
    "#)
}

/// Ruta GET /health: Diagnóstico y versión de los pesos.
async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let (version, pesos) = {
        let m = state.modelo.read().unwrap();
        (m.version, m.pesos.clone())
    };

    let total = state.total_consultas.load(Ordering::Relaxed);

    println!("  [HTTP] 🟢 GET /health atendido.");

    Json(HealthResponse {
        status: "ok".to_string(),
        modelo_version: version,
        pesos_actuales: pesos,
        total_predicciones_servidas: total,
    })
}

/// Ruta POST /predict: Inferencia concurrente con RwLock.
async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    state.total_consultas.fetch_add(1, Ordering::Relaxed);

    let (pred, version) = {
        let m = state.modelo.read().unwrap();
        (m.predecir(&payload.features), m.version)
    };

    println!(
        "  [HTTP] 🎯 POST /predict: entrada {:?} -> Predicción: {:.4} (v{})",
        payload.features, pred, version
    );

    (
        StatusCode::OK,
        Json(PredictResponse {
            prediccion: pred,
            dimension_entrada: payload.features.len(),
            modelo_version: version,
        }),
    )
}

/// Ruta POST /update: Recarga de modelo en caliente (exclusión de escritura).
async fn update_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<UpdateModelRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (nueva_version, nuevos_pesos) = {
        let mut m = state.modelo.write().unwrap();
        m.version += 1;
        m.pesos = payload.nuevos_pesos.clone();
        (m.version, m.pesos.clone())
    };

    println!(
        "  [HTTP] 🔄 POST /update: ¡Modelo actualizado en caliente a v{} con pesos {:?}!",
        nueva_version, nuevos_pesos
    );

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "mensaje": "Pesos actualizados con éxito",
            "nueva_version": nueva_version,
            "nuevos_pesos": nuevos_pesos
        })),
    )
}

// =============================================================================
// MAIN PERMANENTE
// =============================================================================

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🚀 SERVIDOR DE INFERENCIA EN PRODUCCIÓN (AXUM + TOKIO)");
    println!("============================================================\n");

    let estado = Arc::new(AppState {
        modelo: Arc::new(RwLock::new(ModeloIA {
            version: 1,
            pesos: vec![0.5, 2.0, -1.0], // y = 0.5*x0 + 2.0*x1 - 1.0*x2
        })),
        total_consultas: Arc::new(AtomicUsize::new(0)),
    });

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .route("/update", post(update_handler))
        .with_state(estado);

    let host = "127.0.0.1:3000";
    let listener = TcpListener::bind(host).await.unwrap();

    println!("✅ Servidor escuchando en: http://{}", host);
    println!("\n💡 Puedes probarlo desde otra terminal o tu navegador:");
    println!("   1. Abrir en el navegador: http://127.0.0.1:3000");
    println!("   2. Health check con curl: curl http://127.0.0.1:3000/health");
    println!("   3. Predicción con curl  : curl -X POST http://127.0.0.1:3000/predict \\");
    println!("                             -H \"Content-Type: application/json\" \\");
    println!("                             -d '{{\"features\": [10.0, 5.0, 2.0]}}'");
    println!("   4. Actualizar en caliente: curl -X POST http://127.0.0.1:3000/update \\");
    println!("                             -H \"Content-Type: application/json\" \\");
    println!("                             -d '{{\"nuevos_pesos\": [1.0, 1.0, 1.0]}}'");
    println!("\nPresiona [Ctrl + C] para apagar el servidor cuando desees.\n");

    axum::serve(listener, app).await.unwrap();
}
