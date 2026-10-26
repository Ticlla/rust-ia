//! ============================================================================
//! 🏆 FASE 2 - CLASE 10: PROYECTO FINAL INTEGRADOR DE SISTEMAS Y CONCURRENCIA
//! ============================================================================
//!
//! Este proyecto integra las 9 clases de la Fase 2 en un motor de inferencia de IA:
//! 1. Buffer de memoria contigua de bajo nivel con punteros y `Drop` (Clase 09).
//! 2. Modelo en memoria compartida con `Arc<RwLock>` y Hot-Reloading (Clases 03 y 04).
//! 3. Inferencia tensorial paralela con `Rayon` (Clase 05).
//! 4. Orquestador asíncrono con `Tokio` y `spawn_blocking` (Clases 06 y 07).
//! 5. Servidor web HTTP REST con `Axum` y `serde` (Clase 08).
//! 6. DataLoader en segundo plano con canales `tokio::sync::mpsc` (Clase 02).
//!
//! Para ejecutar la demostración completa:
//!     cargo run --bin c10_proyecto

#![allow(dead_code)]

use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

// =============================================================================
// SUBSISTEMA 1: MEMORIA DE BAJO NIVEL (CLASE 09 - SAFE WRAPPER & RAII)
// =============================================================================

/// Búfer de memoria contigua para pesos de modelos de IA con asignación manual.
#[derive(Debug)]
pub struct TensorBuffer {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

// Permitimos enviar el tensor entre hilos de forma segura:
unsafe impl Send for TensorBuffer {}
unsafe impl Sync for TensorBuffer {}

impl TensorBuffer {
    pub fn from_slice(datos: &[f32]) -> Self {
        assert!(!datos.is_empty(), "El tensor no puede estar vacío");
        let layout = Layout::array::<f32>(datos.len()).expect("Layout inválido");
        let ptr = unsafe { alloc(layout) as *mut f32 };

        if ptr.is_null() {
            panic!("Fallo al asignar memoria física para el tensor");
        }

        unsafe {
            std::ptr::copy_nonoverlapping(datos.as_ptr(), ptr, datos.len());
        }

        Self {
            ptr,
            len: datos.len(),
            layout,
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for TensorBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

// =============================================================================
// SUBSISTEMA 2: MODELO DE IA Y CÓMPUTO PARALELO (CLASES 04 Y 05 - RAYON)
// =============================================================================

#[derive(Debug)]
pub struct ModeloInferencia {
    pub version: usize,
    pub pesos: TensorBuffer,
}

impl ModeloInferencia {
    /// Inferencia paralela con Rayon (aprovecha los 8 núcleos del CPU): y = sum(w * x)
    pub fn predecir_paralelo(&self, entrada: &[f32]) -> f32 {
        let pesos = self.pesos.as_slice();
        assert_eq!(pesos.len(), entrada.len(), "Dimensiones incompatibles");

        // 💡 Rayon divide el cálculo vectorial entre todos los núcleos disponibles:
        pesos
            .par_iter()
            .zip(entrada.par_iter())
            .map(|(&w, &x)| w * x)
            .sum()
    }
}

// =============================================================================
// SUBSISTEMA 3: ESTADO GLOBAL DEL SERVIDOR (CLASES 03 Y 08)
// =============================================================================

pub struct AppState {
    pub modelo: Arc<RwLock<ModeloInferencia>>,
    pub total_inferencias: Arc<AtomicUsize>,
    pub dataloader_tx: mpsc::Sender<String>,
}

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub modelo_version: usize,
    pub dimension_pesos: usize,
    pub total_inferencias_servidas: usize,
}

#[derive(Deserialize, Serialize)]
pub struct PredictRequest {
    pub features: Vec<f32>,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub prediccion: f32,
    pub modelo_version: usize,
}

#[derive(Deserialize)]
pub struct ReloadRequest {
    pub nuevos_pesos: Vec<f32>,
}

// =============================================================================
// SUBSISTEMA 4: HANDLERS WEB DE AXUM (CLASES 07 Y 08)
// =============================================================================

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let (version, dim) = {
        let m = state.modelo.read().unwrap();
        (m.version, m.pesos.len())
    };

    Json(HealthResponse {
        status: "ok".to_string(),
        modelo_version: version,
        dimension_pesos: dim,
        total_inferencias_servidas: state.total_inferencias.load(Ordering::Relaxed),
    })
}

pub async fn predict_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PredictRequest>,
) -> (StatusCode, Json<PredictResponse>) {
    let req_id = state.total_inferencias.fetch_add(1, Ordering::Relaxed) + 1;

    // 💡 REGLA DE ORO DE IA (Clase 07):
    // Desacoplamos el cálculo numérico pesado a 'spawn_blocking' para no congelar la red:
    let modelo_clon = state.modelo.clone();
    let features = payload.features;

    let (pred, version) = tokio::task::spawn_blocking(move || {
        let m = modelo_clon.read().unwrap();
        let resultado = m.predecir_paralelo(&features);
        (resultado, m.version)
    })
    .await
    .unwrap();

    println!("  ⚡ [Inferencia #{req_id:03}] Modelo v{version} | Predicción: {pred:.2}");

    (
        StatusCode::OK,
        Json(PredictResponse {
            prediccion: pred,
            modelo_version: version,
        }),
    )
}

pub async fn hot_reload_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ReloadRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let nuevo_buffer = TensorBuffer::from_slice(&payload.nuevos_pesos);

    // 💡 Modificación segura con exclusión de escritura (RwLock write):
    let nueva_version = {
        let mut m = state.modelo.write().unwrap();
        m.version += 1;
        m.pesos = nuevo_buffer;
        m.version
    };

    println!("\n  🔥 [HOT-RELOAD EXITOSO] ¡Pesos actualizados en caliente a v{nueva_version}!");
    println!("     Nuevos pesos activos: {:?}", payload.nuevos_pesos);
    println!("     (El buffer anterior v{} fue liberado de la RAM con Drop)\n", nueva_version - 1);

    // Notificamos al DataLoader asíncrono sobre la recarga de pesos:
    let _ = state.dataloader_tx.send(format!("Pesos actualizados a v{nueva_version}")).await;

    (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "mensaje": "Hot-Reload completado con éxito",
            "nueva_version": nueva_version,
            "dimension": payload.nuevos_pesos.len()
        })),
    )
}

pub fn crear_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/predict", post(predict_handler))
        .route("/hot_reload", post(hot_reload_handler))
        .with_state(state)
}

// =============================================================================
// MAIN: DEMOSTRACIÓN COMPLETA AUTOMATIZADA
// =============================================================================

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("🏆 PROYECTO FINAL FASE 2: Motor de Inferencia y Servidor IA");
    println!("============================================================\n");

    // 1. Canal asíncrono para el DataLoader en segundo plano (Clase 02 & 07):
    let (data_tx, mut data_rx) = mpsc::channel::<String>(10);

    // 2. Tarea en segundo plano del DataLoader / Ingestor:
    tokio::spawn(async move {
        while let Some(msg) = data_rx.recv().await {
            println!("  [DataLoader Worker] 📦 Evento recibido: {msg}");
        }
    });

    // 3. Inicializamos los pesos del modelo con nuestro TensorBuffer seguro (Clase 09):
    let pesos_iniciales = vec![0.5, 2.0, -1.0]; // y = 0.5*x0 + 2.0*x1 - 1.0*x2
    let estado = Arc::new(AppState {
        modelo: Arc::new(RwLock::new(ModeloInferencia {
            version: 1,
            pesos: TensorBuffer::from_slice(&pesos_iniciales),
        })),
        total_inferencias: Arc::new(AtomicUsize::new(0)),
        dataloader_tx: data_tx,
    });

    // 4. Arrancamos el servidor Axum en un puerto dinámico:
    let app = crear_router(estado.clone());
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    println!("🌐 Servidor de Inferencia activo en: http://{}", addr);

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(50)).await;

    // =========================================================================
    // SIMULACIÓN CLIENTE: PETICIONES CONCURRENTES EN VIVO
    // =========================================================================
    println!("\n--- 1. Verificando Health Check Inicial (/health) ---");
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let get_req = format!("GET /health HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    stream.write_all(get_req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    println!("Respuesta /health:\n{}", String::from_utf8_lossy(&buf));

    println!("--- 2. Enviando Inferencia POST /predict con vector [10.0, 5.0, 2.0] ---");
    let req_body = serde_json::to_string(&PredictRequest {
        features: vec![10.0, 5.0, 2.0],
    })
    .unwrap();
    let post_req = format!(
        "POST /predict HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        req_body.len(),
        req_body
    );
    let mut stream2 = TcpStream::connect(addr).await.unwrap();
    stream2.write_all(post_req.as_bytes()).await.unwrap();
    let mut buf2 = Vec::new();
    stream2.read_to_end(&mut buf2).await.unwrap();
    println!("Respuesta /predict v1:\n{}", String::from_utf8_lossy(&buf2));
    println!("Esperado v1: 10*0.5 + 5*2.0 - 2*1.0 = 13.0");

    println!("--- 3. Ejecutando Hot-Reload en Caliente (/hot_reload) con [1.0, 1.0, 1.0] ---");
    let reload_body = serde_json::json!({ "nuevos_pesos": [1.0, 1.0, 1.0] }).to_string();
    let reload_req = format!(
        "POST /hot_reload HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        reload_body.len(),
        reload_body
    );
    let mut stream3 = TcpStream::connect(addr).await.unwrap();
    stream3.write_all(reload_req.as_bytes()).await.unwrap();
    let mut buf3 = Vec::new();
    stream3.read_to_end(&mut buf3).await.unwrap();
    println!("Respuesta /hot_reload:\n{}", String::from_utf8_lossy(&buf3));

    tokio::time::sleep(Duration::from_millis(50)).await;

    println!("--- 4. Repitiendo Inferencia con el Modelo Actualizado (v2) ---");
    let mut stream4 = TcpStream::connect(addr).await.unwrap();
    stream4.write_all(post_req.as_bytes()).await.unwrap();
    let mut buf4 = Vec::new();
    stream4.read_to_end(&mut buf4).await.unwrap();
    println!("Respuesta /predict v2:\n{}", String::from_utf8_lossy(&buf4));
    println!("Esperado v2: 10*1.0 + 5*1.0 + 2*1.0 = 17.0");

    println!("\n🎉 ¡PROYECTO FINAL FASE 2 COMPLETADO CON ÉXITO!");
}
