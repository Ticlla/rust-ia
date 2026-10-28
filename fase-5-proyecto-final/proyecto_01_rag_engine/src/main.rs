//! Entrypoint del Servidor Enterprise Hybrid RAG Engine

use candle_core::Device;
use proyecto_01_rag_engine::api::handlers::{build_router, AppState};
use proyecto_01_rag_engine::config::RagConfig;
use proyecto_01_rag_engine::engine::EnterpriseRagEngine;
use proyecto_01_rag_engine::ingestion::Document;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Inicializar sistema de logs y trazabilidad
    tracing_subscriber::fmt::init();

    println!("==============================================================================");
    println!("🚀 INICIANDO ENTERPRISE HYBRID RAG ENGINE (RUST + AXUM + CANDLE + SIMD)");
    println!("==============================================================================");

    let config = RagConfig::default();
    let dev = Device::Cpu;

    println!("[1] Configurando Motor RAG:");
    println!("    - Dimensión de embeddings: {}", config.embedding_dim);
    println!("    - Tamaño de chunk: {} caracteres", config.chunk_size);
    println!("    - Overlap: {} caracteres", config.chunk_overlap);
    println!("    - Constante RRF (k): {}", config.rrf_k);
    println!("    - Dispositivo: {:?}", dev);

    let mut engine = EnterpriseRagEngine::new(config.clone(), &dev)?;

    // Precargar un corpus empresarial de prueba
    println!("\n[2] Precargando Base de Conocimiento Empresarial:");
    let corpus = vec![
        Document::new(
            "doc_rust_concurrency",
            "Concurrencia en Rust y Tokio",
            "Rust garantiza seguridad en concurrencia sin data races gracias a su sistema de tipos y ownership. \
             Tokio proporciona un runtime asíncrono basado en un event loop multi-hilo con robo de trabajo (work stealing). \
             Los canales MPSC permiten comunicación eficiente entre tareas asíncronas sin necesidad de bloqueos con mutex.",
        ).with_metadata("category", "systems"),

        Document::new(
            "doc_kv_cache",
            "Optimización de LLMs con KV-Cache",
            "El KV-Cache es la técnica fundamental para reducir la complejidad temporal en inferencia autorregresiva de transformers. \
             Sin KV-Cache, el cálculo es cuadrático O(N^2) porque requiere recalcular todas las representaciones previas. \
             Con KV-Cache, las matrices Key y Value se almacenan en VRAM permitiendo que cada nuevo token requiera solo tiempo O(1).",
        ).with_metadata("category", "ai_serving"),

        Document::new(
            "doc_simd_hardware",
            "Aceleración de Hardware con Intrínsecos SIMD AVX2",
            "Las instrucciones SIMD permiten ejecutar una misma operación matemática sobre múltiples números en un solo ciclo de reloj. \
             En x86_64, AVX2 procesa 8 números flotantes de 32 bits a la vez. \
             Las operaciones FMA (Fused Multiply-Add) aceleran el cálculo del producto punto y la multiplicación matricial en deep learning.",
        ).with_metadata("category", "hardware"),

        Document::new(
            "doc_safetensors_format",
            "Almacenamiento de Modelos con Safetensors y Zero-Copy",
            "El formato Safetensors de Hugging Face previene vulnerabilidades de ejecución remota de código presentes en pickle. \
             Utiliza una cabecera JSON seguida de buffers binarios contiguos alineados a 8 bytes. \
             Combinado con mmap (memory mapping), permite cargar pesos de modelos gigantes a velocidad de bus NVMe con cero copia en RAM.",
        ).with_metadata("category", "storage"),
    ];

    let chunks_created = engine.ingest_documents(corpus)?;
    println!("    -> 4 Documentos empresariales indexados exitosamente.");
    println!("    -> {} Chunks generados en memoria.", chunks_created);

    let state = Arc::new(AppState::new(Arc::new(RwLock::new(engine))));

    let app = build_router(state);
    let addr = format!("0.0.0.0:{}", config.server_port);
    let listener = TcpListener::bind(&addr).await?;

    println!("\n[3] Servidor HTTP listo y escuchando en: http://{}", addr);
    println!("    - GET  /health           -> Verificación de salud y estadísticas");
    println!("    - GET  /metrics          -> Exportador de métricas OpenMetrics / Prometheus");
    println!("    - GET  /api/v1/stats     -> Métricas del índice");
    println!("    - POST /api/v1/documents -> Ingesta dinámica de documentos");
    println!("    - POST /api/v1/search    -> Búsqueda híbrida (Dense, Sparse, Hybrid, Rerank)");
    println!("==============================================================================");

    axum::serve(listener, app).await?;
    Ok(())
}
