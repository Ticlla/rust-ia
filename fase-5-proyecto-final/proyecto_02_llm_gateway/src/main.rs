//! Entrypoint del Servidor LLM Serving Gateway con Soporte para Pesos Reales SmolLM2-135M

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use proyecto_02_llm_gateway::api::handlers::{build_router, AppState};
use proyecto_02_llm_gateway::api::openai_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelCard,
    ModelListResponse, UsageInfo,
};
use proyecto_02_llm_gateway::config::GatewayConfig;
use proyecto_02_llm_gateway::model::{GatewayTokenizer, GatewayTransformer};
use proyecto_02_llm_gateway::paged_cache::PagedKvCacheManager;
use proyecto_02_llm_gateway::scheduler::ContinuousBatchScheduler;
use std::path::Path;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

struct RealServerState {
    model: Arc<Llama>,
    tokenizer: Tokenizer,
    config: LlamaConfig,
    dev: Device,
    model_id: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let model_dir = "/home/alcidesticlla/Documents/rust-ia/models/smollm2-135m";
    let weights_path = format!("{}/model.safetensors", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8081);
    let addr = format!("0.0.0.0:{}", port);

    if Path::new(&weights_path).exists() && Path::new(&tokenizer_path).exists() {
        println!("==============================================================================");
        println!("🚀 INICIANDO LLM SERVING GATEWAY CON MODELO REAL (SMOLLM2-135M) EN RUST");
        println!("==============================================================================");
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

        println!("[1] Mapeando pesos Safetensors (257 MB) con Zero-Copy (mmap)...");
        let start_load = Instant::now();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &dev)? };
        let model = Arc::new(Llama::load(vb, &llama_config)?);
        println!("    -> ¡Modelo cargado en {:.2} ms!", start_load.elapsed().as_secs_f64() * 1000.0);

        println!("[2] Cargando tokenizador BPE Hugging Face (49,152 tokens)...");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

        let server_state = Arc::new(RealServerState {
            model,
            tokenizer,
            config: llama_config,
            dev,
            model_id: "smollm2-135m".to_string(),
        });

        let app = Router::new()
            .route("/health", get(real_health_handler))
            .route("/v1/models", get(real_models_handler))
            .route("/v1/chat/completions", post(real_chat_handler))
            .with_state(server_state);

        println!("\n[3] Gateway Real OpenAI-Compatible activo en: http://{}", addr);
        println!("    - GET  /health               -> Monitor de salud del modelo");
        println!("    - GET  /v1/models            -> Especificación de modelos cargados");
        println!("    - POST /v1/chat/completions  -> Inferencia real con SmolLM2");
        println!("==============================================================================");

        let listener = TcpListener::bind(&addr).await?;
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await?;
        println!("✅ [LLM GATEWAY] Servidor apagado de forma limpia.");
        return Ok(());
    }

    // Fallback: Si no existen los pesos en disco, correr el modo mock de Continuous Batching
    println!("==============================================================================");
    println!("⚡ INICIANDO HIGH-THROUGHPUT LLM SERVING GATEWAY (MODO CONTINUOUS BATCHING MOCK)");
    println!("==============================================================================");

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

    let state = Arc::new(AppState::new(config.clone(), scheduler));

    let app = build_router(state);
    let listener = TcpListener::bind(&addr).await?;
    println!("\n[3] Gateway Mock activo en: http://{}", addr);
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

async fn real_health_handler(State(state): State<Arc<RealServerState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "model": state.model_id,
        "engine": "candle_llama_real_weights",
        "parameters": "135M",
        "vocabulary_size": 49152,
    }))
}

async fn real_models_handler(State(state): State<Arc<RealServerState>>) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.model_id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "huggingface-smollm2".to_string(),
        }],
    })
}

async fn real_chat_handler(
    State(state): State<Arc<RealServerState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let mut prompt = String::new();
    for msg in &payload.messages {
        prompt.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", msg.role, msg.content));
    }
    prompt.push_str("<|im_start|>assistant\n");

    let encoded = state
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Error de tokenización: {e}")))?;
    let prompt_ids = encoded.get_ids().to_vec();
    let prompt_len = prompt_ids.len();

    let model = Arc::clone(&state.model);
    let config = state.config.clone();
    let dev = state.dev.clone();
    let tokenizer = state.tokenizer.clone();
    let max_tokens = payload.max_tokens.min(96);
    let temperature = payload.temperature.max(0.01) as f64;

    // Ejecutar la inferencia neuronal en un hilo de trabajo dedicado (spawn_blocking)
    let generated_text = tokio::task::spawn_blocking(move || -> Result<String, String> {
        let mut cache = Cache::new(true, DType::F32, &config, &dev).map_err(|e| e.to_string())?;

        let input_tensor = Tensor::new(&prompt_ids[..], &dev)
            .map_err(|e| e.to_string())?
            .unsqueeze(0)
            .map_err(|e| e.to_string())?;

        let logits = model
            .forward(&input_tensor, 0, &mut cache)
            .map_err(|e| e.to_string())?;

        let last_logits = logits
            .squeeze(0)
            .map_err(|e| e.to_string())?
            .squeeze(0)
            .map_err(|e| e.to_string())?;

        let mut next_token = sample_token(&last_logits, temperature).map_err(|e| e.to_string())?;

        let mut generated_ids = Vec::new();
        let eos_id = 2u32;
        let im_end_id = 49153u32;

        for step in 0..max_tokens {
            if next_token == eos_id || next_token == im_end_id {
                break;
            }
            generated_ids.push(next_token);

            let input_token = Tensor::new(&[next_token], &dev)
                .map_err(|e| e.to_string())?
                .unsqueeze(0)
                .map_err(|e| e.to_string())?;

            let logits = model
                .forward(&input_token, prompt_len + step, &mut cache)
                .map_err(|e| e.to_string())?;

            let token_logits = logits
                .squeeze(0)
                .map_err(|e| e.to_string())?
                .squeeze(0)
                .map_err(|e| e.to_string())?;

            next_token = sample_token(&token_logits, temperature).map_err(|e| e.to_string())?;
        }

        tokenizer.decode(&generated_ids, false).map_err(|e| e.to_string())
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Error en join: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Error en inferencia: {e}")))?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let response = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: now,
        model: state.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: generated_text,
            },
            finish_reason: Some("stop".to_string()),
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_len,
            completion_tokens: 0,
            total_tokens: prompt_len,
        },
    };

    Ok(Json(response).into_response())
}

fn sample_token(logits: &Tensor, temperature: f64) -> candle_core::Result<u32> {
    if temperature <= 0.001 {
        let vals = logits.to_vec1::<f32>()?;
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in vals.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }

    let scaled = (logits / temperature)?;
    let probs = candle_nn::ops::softmax(&scaled, 0)?;
    let vals = probs.to_vec1::<f32>()?;

    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in vals.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    Ok(max_idx as u32)
}
