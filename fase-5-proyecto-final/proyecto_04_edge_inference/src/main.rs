//! Servidor principal del microservicio Edge Vision & Audio.

use std::net::SocketAddr;
use std::path::Path;
use std::time::Duration;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use proyecto_04_edge_inference::api::{create_router, AppState};
use proyecto_04_edge_inference::config::EdgeServiceConfig;
use proyecto_04_edge_inference::engine::SessionPool;
use proyecto_04_edge_inference::queue::{BackpressureQueue, InferenceBatcher};

// Modelos binarios por defecto incluidos en compilación para portabilidad total
const DEFAULT_VISION_MODEL: &[u8] = include_bytes!("../models/vision_classifier.onnx");
const DEFAULT_AUDIO_MODEL: &[u8] = include_bytes!("../models/audio_classifier.onnx");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let config = EdgeServiceConfig::default();

    println!("================================================================================");
    println!("  👁️  EDGE VISION & AUDIO REAL-TIME INFERENCE MICROSERVICE (PROYECTO 04)");
    println!("================================================================================");
    println!("  Configuración activa:");
    println!("    • Host: {}", config.host);
    println!("    • Puerto: {}", config.port);
    println!("    • Pool de sesiones ONNX: {} workers por modalidad", config.pool_size);
    println!("    • Capacidad de cola (Backpressure): {} slots", config.queue_capacity);
    println!("    • Dynamic Batching: Máx {} muestras / {} ms ventana", config.max_batch_size, config.batch_timeout_ms);
    println!("--------------------------------------------------------------------------------");

    // 1. Cargar modelos ONNX desde disco o memoria embebida
    let vision_bytes = if Path::new(&config.vision_model_path).exists() {
        std::fs::read(&config.vision_model_path)?
    } else {
        DEFAULT_VISION_MODEL.to_vec()
    };

    let audio_bytes = if Path::new(&config.audio_model_path).exists() {
        std::fs::read(&config.audio_model_path)?
    } else {
        DEFAULT_AUDIO_MODEL.to_vec()
    };

    // 2. Inicializar Pool de Sesiones ONNX Runtime
    println!("⚡ Inicializando pool de sesiones de inferencia ONNX...");
    let pool = SessionPool::new(&config, &vision_bytes, &audio_bytes)?;
    println!("✓ Pool ONNX preparado con éxito.");

    // 3. Crear Cola de Inferencia con Contrapresión
    let (queue, receiver) = BackpressureQueue::new(config.queue_capacity);

    // 4. Iniciar Worker de Dynamic Batching en segundo plano
    let batch_timeout = Duration::from_millis(config.batch_timeout_ms);
    let _worker_handle = InferenceBatcher::spawn(receiver, pool.clone(), config.max_batch_size, batch_timeout);
    println!("✓ Worker de Dynamic Batching en ejecución.");

    // 5. Configurar Estado Global y Router Axum
    let state = AppState::new(queue, pool);
    let app = create_router(state);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    println!("🚀 Servidor Edge escuchando en: http://{}", addr);
    println!("   • Visión: POST http://{}/v1/vision/classify", addr);
    println!("   • Audio:  POST http://{}/v1/audio/infer", addr);
    println!("   • Health: GET  http://{}/v1/health", addr);
    println!("   • Mtrcs:  GET  http://{}/v1/metrics", addr);
    println!("================================================================================\n");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
