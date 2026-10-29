//! Manejadores de Rutas de la API OpenAI-Compatible

use super::openai_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelCard,
    ModelListResponse, UsageInfo,
};
use super::sse::create_sse_response;
use crate::config::GatewayConfig;
use crate::resilience::TokenBucketRateLimiter;
use crate::scheduler::{ContinuousBatchScheduler, FinishReason, Sequence};
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse, Response},
    routing::{get, post},
};
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use chrono::Utc;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use uuid::Uuid;

pub struct RealLlamaEngine {
    pub model: Arc<Llama>,
    pub tokenizer: Tokenizer,
    pub config: LlamaConfig,
    pub dev: Device,
    pub model_id: String,
}

pub struct AppState {
    pub config: GatewayConfig,
    pub scheduler: Arc<ContinuousBatchScheduler>,
    pub real_engine: Option<Arc<RealLlamaEngine>>,
    pub requests_total: AtomicU64,
    pub rag_retrievals_total: AtomicU64,
    pub rate_limited_total: AtomicU64,
    pub rate_limiter: Arc<TokenBucketRateLimiter>,
}

impl AppState {
    pub fn new(config: GatewayConfig, scheduler: Arc<ContinuousBatchScheduler>) -> Self {
        Self::with_engine(config, scheduler, None)
    }

    pub fn with_engine(
        config: GatewayConfig,
        scheduler: Arc<ContinuousBatchScheduler>,
        real_engine: Option<Arc<RealLlamaEngine>>,
    ) -> Self {
        Self {
            config,
            scheduler,
            real_engine,
            requests_total: AtomicU64::new(0),
            rag_retrievals_total: AtomicU64::new(0),
            rate_limited_total: AtomicU64::new(0),
            // Ráfaga permitida de 10 peticiones, recarga de 2 peticiones/segundo
            rate_limiter: Arc::new(TokenBucketRateLimiter::new(10.0, 2.0)),
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(chat_ui_handler))
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

pub fn ship_log(
    service: &'static str,
    level: &'static str,
    event: &'static str,
    action: &'static str,
    status_code: u16,
    latency_ms: f64,
    client_ip: &str,
    detail: &str,
) {
    let es_url = std::env::var("ELASTICSEARCH_URL")
        .unwrap_or_else(|_| "http://elasticsearch:9200".to_string());
    let target = format!("{}/ai-stack-logs/_doc", es_url.trim_end_matches('/'));

    let payload = serde_json::json!({
        "@timestamp": Utc::now().to_rfc3339(),
        "service": service,
        "level": level,
        "event": event,
        "action": action,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "client_ip": client_ip,
        "detail": detail,
        "env": "production"
    });

    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(800))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let _ = client.post(&target).json(&payload).send().await;
    });
}

async fn chat_ui_handler() -> Html<&'static str> {
    Html(include_str!("chat_ui.html"))
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cache = state.scheduler.paged_cache.lock().await;
    let waiting = state.scheduler.waiting_queue.lock().await;
    let running = state.scheduler.running_sequences.lock().await;

    let (engine_type, model_id) = match &state.real_engine {
        Some(e) => ("candle_llama_real_weights", e.model_id.as_str()),
        None => ("continuous_batching_mock", state.config.model_id.as_str()),
    };

    Json(serde_json::json!({
        "status": "healthy",
        "engine": engine_type,
        "model": model_id,
        "waiting_requests": waiting.len(),
        "running_requests": running.len(),
        "memory_utilization": cache.allocator.memory_utilization(),
        "free_blocks": cache.allocator.num_free_blocks(),
        "used_blocks": cache.allocator.num_used_blocks(),
    }))
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cache = state.scheduler.paged_cache.lock().await;
    let waiting = state.scheduler.waiting_queue.lock().await;
    let running = state.scheduler.running_sequences.lock().await;

    let requests = state.requests_total.load(Ordering::Relaxed);
    let rag_retrievals = state.rag_retrievals_total.load(Ordering::Relaxed);
    let rate_limited = state.rate_limited_total.load(Ordering::Relaxed);
    let mem_util = cache.allocator.memory_utilization();
    let free_blocks = cache.allocator.num_free_blocks();
    let used_blocks = cache.allocator.num_used_blocks();
    let waiting_count = waiting.len();
    let running_count = running.len();

    let body = format!(
        "# HELP llm_gateway_requests_total Total de peticiones HTTP procesadas por el Gateway\n\
         # TYPE llm_gateway_requests_total counter\n\
         llm_gateway_requests_total {}\n\n\
         # HELP llm_gateway_rag_retrievals_total Peticiones donde se recuperó contexto de RAG\n\
         # TYPE llm_gateway_rag_retrievals_total counter\n\
         llm_gateway_rag_retrievals_total {}\n\n\
         # HELP llm_gateway_rate_limited_total Peticiones bloqueadas por superar el límite Token Bucket (HTTP 429)\n\
         # TYPE llm_gateway_rate_limited_total counter\n\
         llm_gateway_rate_limited_total {}\n\n\
         # HELP llm_gateway_kv_cache_utilization_ratio Ratio de utilización del Paged KV-Cache\n\
         # TYPE llm_gateway_kv_cache_utilization_ratio gauge\n\
         llm_gateway_kv_cache_utilization_ratio {:.4}\n\n\
         # HELP llm_gateway_kv_cache_free_blocks Bloques libres en el gestor de memoria Paged KV-Cache\n\
         # TYPE llm_gateway_kv_cache_free_blocks gauge\n\
         llm_gateway_kv_cache_free_blocks {}\n\n\
         # HELP llm_gateway_kv_cache_used_blocks Bloques en uso en el gestor de memoria Paged KV-Cache\n\
         # TYPE llm_gateway_kv_cache_used_blocks gauge\n\
         llm_gateway_kv_cache_used_blocks {}\n\n\
         # HELP llm_gateway_waiting_requests Peticiones esperando en cola del scheduler\n\
         # TYPE llm_gateway_waiting_requests gauge\n\
         llm_gateway_waiting_requests {}\n\n\
         # HELP llm_gateway_running_requests Secuencias activas ejecutándose concurrentemente\n\
         # TYPE llm_gateway_running_requests gauge\n\
         llm_gateway_running_requests {}\n",
        requests,
        rag_retrievals,
        rate_limited,
        mem_util,
        free_blocks,
        used_blocks,
        waiting_count,
        running_count,
    );

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
}

async fn models_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let model_id = state
        .real_engine
        .as_ref()
        .map(|e| e.model_id.clone())
        .unwrap_or_else(|| state.config.model_id.clone());

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: model_id,
            object: "model".to_string(),
            created: now,
            owned_by: "rust-ia-systems".to_string(),
        }],
    })
}

fn sample_token_local(logits: &Tensor, temperature: f64) -> candle_core::Result<u32> {
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

fn format_chatml_prompt(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            msg.role, msg.content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    let start_time = Instant::now();
    state.requests_total.fetch_add(1, Ordering::Relaxed);
    let client_ip = "127.0.0.1";

    // 1. Escudo de Resiliencia: Control de admisión con Rate-Limiting (Token Bucket)
    let client_id = "default_client";
    if let Err(retry_after) = state.rate_limiter.check_admission(client_id) {
        state.rate_limited_total.fetch_add(1, Ordering::Relaxed);
        let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        ship_log(
            "ai-llm-gateway",
            "WARN",
            "rate_limit_exceeded",
            "rate_limiter_evaluation",
            429,
            latency_ms,
            client_ip,
            &format!("Token Bucket agotado. Reintente en {:.1}s", retry_after),
        );

        let err_json = serde_json::json!({
            "error": {
                "message": format!("Rate limit excedido (Token Bucket). Reintente en {:.1} segundos.", retry_after),
                "type": "rate_limit_exceeded",
                "code": 429
            }
        });
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [(
                axum::http::header::RETRY_AFTER,
                format!("{:.0}", retry_after),
            )],
            Json(err_json),
        )
            .into_response());
    }

    let mut messages = payload.messages.clone();

    // 2. Pipeline RAG Automático: Si RAG_SERVICE_URL está presente, recuperar contexto por la red interna
    if let Ok(rag_url) = std::env::var("RAG_SERVICE_URL") {
        if !rag_url.trim().is_empty() {
            if let Some(user_msg) = payload.messages.iter().rev().find(|m| m.role == "user") {
                let rag_client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_millis(1500))
                    .build()
                    .unwrap_or_else(|_| reqwest::Client::new());
                let search_body = serde_json::json!({
                    "query": user_msg.content,
                    "top_k": 2
                });
                let search_url = format!("{}/api/v1/search", rag_url.trim_end_matches('/'));

                if let Ok(rag_resp) = rag_client.post(&search_url).json(&search_body).send().await {
                    if rag_resp.status().is_success() {
                        if let Ok(rag_json) = rag_resp.json::<serde_json::Value>().await {
                            if let Some(hits) = rag_json["hits"].as_array() {
                                if !hits.is_empty() {
                                    state.rag_retrievals_total.fetch_add(1, Ordering::Relaxed);
                                    let mut context_text = String::from(
                                        "INFORMACIÓN RECUPERADA DE LA BASE DE CONOCIMIENTO (RAG EN MEMORIA):\n",
                                    );
                                    for (idx, hit) in hits.iter().enumerate() {
                                        let title = hit["doc_title"].as_str().unwrap_or("Documento");
                                        let text = hit["text"].as_str().unwrap_or("");
                                        context_text.push_str(&format!(
                                            "[Fuente {}: {}]\n{}\n\n",
                                            idx + 1,
                                            title,
                                            text
                                        ));
                                    }
                                    context_text.push_str("INSTRUCCIÓN: Responde de forma clara y directa en español utilizando los hechos verificados de arriba.");

                                    messages.insert(
                                        0,
                                        ChatMessage {
                                            role: "system".to_string(),
                                            content: context_text,
                                        },
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 3. Upstream NVIDIA API con Timeout Explícito de 3.5 segundos
    if let Ok(api_key) = std::env::var("NVIDIA_API_KEY") {
        if !api_key.trim().is_empty() {
            let model = std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string());

            let nvidia_client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(3500))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            let max_tokens = if payload.max_tokens == 0 { 256 } else { payload.max_tokens };
            let body = serde_json::json!({
                "model": model,
                "messages": messages,
                "temperature": payload.temperature,
                "max_tokens": max_tokens,
            });

            let nvidia_start = Instant::now();
            let res = nvidia_client
                .post("https://integrate.api.nvidia.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key.trim()))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match res {
                Ok(resp) => {
                    let upstream_status = resp.status();
                    if upstream_status.is_success() {
                        if let Ok(json_body) = resp.json::<serde_json::Value>().await {
                            let total_latency = start_time.elapsed().as_secs_f64() * 1000.0;
                            ship_log(
                                "ai-llm-gateway",
                                "INFO",
                                "chat_completion_success",
                                "upstream_nvidia_nemotron",
                                200,
                                total_latency,
                                client_ip,
                                "Inferencia generada exitosamente con NVIDIA Nemotron",
                            );
                            return Ok(Json(json_body).into_response());
                        }
                    } else {
                        let err_text = resp.text().await.unwrap_or_default();
                        let upstream_latency = nvidia_start.elapsed().as_secs_f64() * 1000.0;
                        ship_log(
                            "ai-llm-gateway",
                            "ERROR",
                            "upstream_nvidia_error",
                            "fallback_to_local",
                            upstream_status.as_u16(),
                            upstream_latency,
                            client_ip,
                            &format!("Upstream error {}: {}", upstream_status, err_text),
                        );
                    }
                }
                Err(e) => {
                    let upstream_latency = nvidia_start.elapsed().as_secs_f64() * 1000.0;
                    ship_log(
                        "ai-llm-gateway",
                        "ERROR",
                        "upstream_nvidia_timeout",
                        "fallback_to_local",
                        504,
                        upstream_latency,
                        client_ip,
                        &format!("Upstream timeout ({:?}). Procediendo a inferencia local.", e),
                    );
                }
            }
        }
    }

    // 4. Inferencia Local: Caso A — Servir modelo real SmolLM2-135M con Candle en CPU
    if let Some(engine) = &state.real_engine {
        let prompt_text = format_chatml_prompt(&messages);
        
        let encoded = engine.tokenizer.encode(prompt_text.as_str(), true).map_err(|e| {
            (StatusCode::BAD_REQUEST, format!("Error de tokenización: {e}"))
        })?;
        let prompt_ids = encoded.get_ids().to_vec();
        let prompt_len = prompt_ids.len();

        let model = Arc::clone(&engine.model);
        let config = engine.config.clone();
        let dev = engine.dev.clone();
        let tokenizer = engine.tokenizer.clone();
        let max_tokens = if payload.max_tokens == 0 { 64 } else { payload.max_tokens.min(96) };
        let temperature = payload.temperature.max(0.01) as f64;

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

            let mut next_token = sample_token_local(&last_logits, temperature).map_err(|e| e.to_string())?;

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

                next_token = sample_token_local(&token_logits, temperature).map_err(|e| e.to_string())?;
            }

            tokenizer
                .decode(&generated_ids, false)
                .map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Error en join: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("Error en inferencia: {e}")))?;

        let total_latency = start_time.elapsed().as_secs_f64() * 1000.0;
        let completion_tokens = max_tokens as usize;

        ship_log(
            "ai-llm-gateway",
            "INFO",
            "local_smollm2_success",
            "self_hosted_serving",
            200,
            total_latency,
            client_ip,
            &format!("Inferencia local SmolLM2 completada en {:.1}ms", total_latency),
        );

        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        let response = ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: now,
            model: engine.model_id.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    content: generated_text.trim().to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: UsageInfo {
                prompt_tokens: prompt_len,
                completion_tokens,
                total_tokens: prompt_len + completion_tokens,
            },
        };

        return Ok(Json(response).into_response());
    }

    // 4. Inferencia Local: Caso B — Fallback a ContinuousBatchScheduler mock
    let mut prompt = String::new();
    for msg in &messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");

    let prompt_tokens = state.scheduler.tokenizer.encode(&prompt).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("Error de tokenización: {}", e),
        )
    })?;

    if prompt_tokens.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Prompt no puede estar vacío".to_string(),
        ));
    }

    let seq_id = format!("chatcmpl-{}", Uuid::new_v4());
    let (tx, mut rx) = mpsc::channel(64);

    let sequence = Sequence::new(
        seq_id.clone(),
        prompt_tokens.clone(),
        payload.temperature,
        payload.max_tokens,
        tx,
    );

    // Enviar secuencia al scheduler de continuous batching
    state.scheduler.add_request(sequence).await;

    // Si el cliente solicitó Streaming vía SSE:
    if payload.stream {
        let sse = create_sse_response(seq_id, state.config.model_id.clone(), rx);
        return Ok(sse.into_response());
    }

    // Si solicitó respuesta síncrona completa:
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let mut full_content = String::new();
    let mut finish_reason_str = "stop".to_string();
    let mut completion_tokens = 0;

    while let Some(chunk) = rx.recv().await {
        full_content.push_str(&chunk.text_delta);
        completion_tokens += 1;
        if let Some(reason) = chunk.finish_reason {
            finish_reason_str = match reason {
                FinishReason::Stop => "stop".to_string(),
                FinishReason::Length => "length".to_string(),
                FinishReason::Abort => "abort".to_string(),
            };
            break;
        }
    }

    let total_latency = start_time.elapsed().as_secs_f64() * 1000.0;
    ship_log(
        "ai-llm-gateway",
        "INFO",
        "local_mock_batch_success",
        "continuous_batch_scheduler",
        200,
        total_latency,
        client_ip,
        &format!("Inferencia mock completada en {:.1}ms ({} tokens)", total_latency, completion_tokens),
    );

    let response = ChatCompletionResponse {
        id: seq_id,
        object: "chat.completion".to_string(),
        created: now,
        model: state.config.model_id.clone(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: full_content.trim().to_string(),
            },
            finish_reason: Some(finish_reason_str),
        }],
        usage: UsageInfo {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens,
            total_tokens: prompt_tokens.len() + completion_tokens,
        },
    };

    Ok(Json(response).into_response())
}
