//! Entrypoint del Servidor LLM Serving Gateway con Soporte para Pesos Reales SmolLM2-135M

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Config as LlamaConfig, Llama};
use proyecto_02_llm_gateway::api::handlers::{AppState, RealLlamaEngine, build_router};
use proyecto_02_llm_gateway::config::GatewayConfig;
use proyecto_02_llm_gateway::model::{GatewayTokenizer, GatewayTransformer};
use proyecto_02_llm_gateway::paged_cache::PagedKvCacheManager;
use proyecto_02_llm_gateway::scheduler::ContinuousBatchScheduler;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::{Duration, sleep};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| {
        if Path::new("/models/smollm2-135m").exists() {
            "/models/smollm2-135m".to_string()
        } else {
            "../../models/smollm2-135m".to_string()
        }
    });

    let weights_path = format!("{}/model.safetensors", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);
    let addr = format!("0.0.0.0:{}", port);

    println!("==============================================================================");
    println!("🚀 INICIANDO HIGH-PERFORMANCE LLM SERVING GATEWAY (RUST + AXUM + CANDLE)");
    println!("==============================================================================");

    let real_engine = if Path::new(&weights_path).exists() && Path::new(&tokenizer_path).exists() {
        println!("[1] Pesos de SmolLM2-135M detectados en: {}", model_dir);
        let dev = Device::Cpu;

        let llama_config = LlamaConfig {
            hidden_size: 576,
            intermediate_size: 1536,
            vocab_size: 49152,
            num_hidden_layers: 30,
            num_attention_heads: 9,
            num_key_value_heads: 3,
            use_flash_attn: false,
            rms_norm_eps: 1e-5,
            rope_theta: 100000.0,
            bos_token_id: Some(1),
            eos_token_id: None,
            rope_scaling: None,
            max_position_embeddings: 8192,
            tie_word_embeddings: true,
        };

        println!("    -> Mapeando pesos Safetensors (257 MB) con Zero-Copy (mmap)...");
        let start_load = Instant::now();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &dev)? };
        let model = Arc::new(Llama::load(vb, &llama_config)?);
        println!(
            "    -> ¡Modelo SmolLM2 mapeado exitosamente en {:.2} ms!",
            start_load.elapsed().as_secs_f64() * 1000.0
        );

        println!("    -> Cargando tokenizador Hugging Face BPE (49,152 tokens)...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        Some(Arc::new(RealLlamaEngine {
            model,
            tokenizer,
            config: llama_config,
            dev,
            model_id: "smollm2-135m".to_string(),
        }))
    } else {
        println!("[1] No se encontraron pesos en {}. Modo Continuous Batching activo.", model_dir);
        None
    };

    println!("[2] Inicializando subsistema de Continuous Batching y Paged KV-Cache...");
    let config = GatewayConfig::default();
    let dev = Device::Cpu;

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

    let scheduler_loop = scheduler.clone();
    tokio::spawn(async move {
        loop {
            match scheduler_loop.step().await {
                Ok(active_count) => {
                    if active_count == 0 {
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

    let state = Arc::new(AppState::with_engine(config, scheduler, real_engine));
    let app = build_router(state);

    println!("[3] Gateway activo y escuchando en: http://{}", addr);
    println!("    - GET  /                     -> Interfaz de Chat Web Interactiva");
    println!("    - GET  /health               -> Monitor de Salud del Gateway");
    println!("    - GET  /metrics              -> Métricas Prometheus");
    println!("    - GET  /v1/models            -> Modelos OpenAI-Compatible");
    println!("    - POST /v1/chat/completions  -> Pipeline Inferencia (RAG + Nvidia/Local + Logs ES)");
    println!("==============================================================================");

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    println!("✅ [LLM GATEWAY] Servidor apagado de forma limpia y segura.");
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Fallo al registrar listener de Ctrl+C");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Fallo al registrar listener de SIGTERM")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            println!("\n🛑 [LLM GATEWAY] Señal Ctrl+C recibida. Drenando conexiones activas...");
        },
        _ = terminate => {
            println!("\n🛑 [LLM GATEWAY] Señal SIGTERM recibida (Docker/K8s). Drenando conexiones activas...");
        },
    }
}
