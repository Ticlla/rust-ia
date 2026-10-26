//! Entrypoint del Servidor LLM Serving Gateway

use candle_core::Device;
use proyecto_02_llm_gateway::api::handlers::{build_router, AppState};
use proyecto_02_llm_gateway::config::GatewayConfig;
use proyecto_02_llm_gateway::model::{GatewayTokenizer, GatewayTransformer};
use proyecto_02_llm_gateway::paged_cache::PagedKvCacheManager;
use proyecto_02_llm_gateway::scheduler::ContinuousBatchScheduler;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    println!("==============================================================================");
    println!("⚡ INICIANDO HIGH-THROUGHPUT LLM SERVING GATEWAY (vLLM EN RUST)");
    println!("==============================================================================");

    let config = GatewayConfig::default();
    let dev = Device::Cpu;

    println!("[1] Inicializando Motor de Inferencia:");
    println!("    - Modelo ID: {}", config.model_id);
    println!("    - Dimensión de embedding: {}", config.embed_dim);
    println!("    - Capas Transformer: {}", config.n_layers);
    println!("    - Tamaño de bloque Paged KV-Cache: {} tokens", config.block_size);
    println!("    - Bloques físicos asignados: {} (Capacidad: {} tokens)", 
        config.total_blocks, config.total_blocks * config.block_size);
    println!("    - Máximo batch continuo concurrente: {}", config.max_batch_size);
    println!("    - Dispositivo: {:?}", dev);

    // Inicializar Componentes
    let model = Arc::new(GatewayTransformer::new(config.clone(), &dev)?);
    let tokenizer = GatewayTokenizer::new_builtin();
    let paged_cache = Arc::new(Mutex::new(PagedKvCacheManager::new(
        config.total_blocks,
        config.block_size,
        config.n_layers,
        config.n_heads,
        config.embed_dim / config.n_heads,
        &dev,
    )));

    let scheduler = Arc::new(ContinuousBatchScheduler::new(
        model,
        tokenizer,
        paged_cache,
        config.max_batch_size,
        config.eos_token_id,
        config.block_size,
    ));

    // [2] Lanzar el Bucle en Segundo Plano de Continuous Batching
    println!("\n[2] Arrancando Bucle Asíncrono de Continuous Batching...");
    let scheduler_loop = scheduler.clone();
    tokio::spawn(async move {
        loop {
            match scheduler_loop.step().await {
                Ok(active_count) => {
                    if active_count == 0 {
                        // Si no hay peticiones activas, reposar brevemente para no saturar CPU en vacío
                        sleep(Duration::from_millis(5)).await;
                    }
                }
                Err(e) => {
                    eprintln!("Error en iteración de Continuous Batching: {}", e);
                    sleep(Duration::from_millis(10)).await;
                }
            }
        }
    });

    let state = Arc::new(AppState {
        config: config.clone(),
        scheduler,
    });

    let app = build_router(state);
    let addr = format!("{}:{}", config.host, config.port);
    let listener = TcpListener::bind(&addr).await?;

    println!("\n[3] Gateway OpenAI-Compatible activo en: http://{}", addr);
    println!("    - GET  /health               -> Monitor de saturación de bloques y colas");
    println!("    - GET  /v1/models            -> Especificación de modelos cargados");
    println!("    - POST /v1/chat/completions  -> Inferencia estándar y Streaming SSE");
    println!("==============================================================================");

    axum::serve(listener, app).await?;
    Ok(())
}
