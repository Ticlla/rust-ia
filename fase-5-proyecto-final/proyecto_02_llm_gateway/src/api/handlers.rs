//! Manejadores de Rutas de la API OpenAI-Compatible

use super::openai_types::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ModelCard,
    ModelListResponse, UsageInfo,
};
use super::sse::create_sse_response;
use crate::config::GatewayConfig;
use crate::scheduler::{ContinuousBatchScheduler, FinishReason, Sequence};
use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use uuid::Uuid;

pub struct AppState {
    pub config: GatewayConfig,
    pub scheduler: Arc<ContinuousBatchScheduler>,
    pub requests_total: AtomicU64,
    pub rag_retrievals_total: AtomicU64,
}

impl AppState {
    pub fn new(config: GatewayConfig, scheduler: Arc<ContinuousBatchScheduler>) -> Self {
        Self {
            config,
            scheduler,
            requests_total: AtomicU64::new(0),
            rag_retrievals_total: AtomicU64::new(0),
        }
    }
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completions_handler))
        .with_state(state)
}

async fn health_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let cache = state.scheduler.paged_cache.lock().await;
    let waiting = state.scheduler.waiting_queue.lock().await;
    let running = state.scheduler.running_sequences.lock().await;

    Json(serde_json::json!({
        "status": "healthy",
        "model": state.config.model_id,
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

    Json(ModelListResponse {
        object: "list".to_string(),
        data: vec![ModelCard {
            id: state.config.model_id.clone(),
            object: "model".to_string(),
            created: now,
            owned_by: "rust-ia-systems".to_string(),
        }],
    })
}

async fn chat_completions_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Result<Response, (StatusCode, String)> {
    state.requests_total.fetch_add(1, Ordering::Relaxed);
    let mut messages = payload.messages.clone();

    // 0. Pipeline RAG Automático: Si RAG_SERVICE_URL está presente, recuperar contexto por la red interna
    if let Ok(rag_url) = std::env::var("RAG_SERVICE_URL") {
        if !rag_url.trim().is_empty() {
            if let Some(user_msg) = payload.messages.iter().rev().find(|m| m.role == "user") {
                let client = reqwest::Client::new();
                let search_body = serde_json::json!({
                    "query": user_msg.content,
                    "top_k": 2
                });
                let search_url = format!("{}/api/v1/search", rag_url.trim_end_matches('/'));

                if let Ok(rag_resp) = client.post(&search_url).json(&search_body).send().await {
                    if rag_resp.status().is_success() {
                        if let Ok(rag_json) = rag_resp.json::<serde_json::Value>().await {
                            if let Some(hits) = rag_json["hits"].as_array() {
                                if !hits.is_empty() {
                                    state.rag_retrievals_total.fetch_add(1, Ordering::Relaxed);
                                    let mut context_text = String::from(
                                        "INFORMACIÓN RECUPERADA DE LA BASE DE CONOCIMIENTO (RAG EN MEMORIA):\n"
                                    );
                                    for (idx, hit) in hits.iter().enumerate() {
                                        let title = hit["doc_title"].as_str().unwrap_or("Documento");
                                        let text = hit["text"].as_str().unwrap_or("");
                                        context_text.push_str(&format!("[Fuente {}: {}]\n{}\n\n", idx + 1, title, text));
                                    }
                                    context_text.push_str("INSTRUCCIÓN: Responde a la pregunta del usuario utilizando como base los hechos verificados del contexto recuperado arriba.");

                                    messages.insert(0, ChatMessage {
                                        role: "system".to_string(),
                                        content: context_text,
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 1. Si existe NVIDIA_API_KEY configurada, reenviar la inferencia a Nemotron con el contexto aumentado
    if let Ok(api_key) = std::env::var("NVIDIA_API_KEY") {
        if !api_key.trim().is_empty() {
            let model = std::env::var("NVIDIA_MODEL")
                .unwrap_or_else(|_| "nvidia/nemotron-3.5-bolt-e4b".to_string());

            let client = reqwest::Client::new();
            let max_tokens = if payload.max_tokens == 0 { 256 } else { payload.max_tokens };
            let body = serde_json::json!({
                "model": model,
                "messages": messages,
                "temperature": payload.temperature,
                "max_tokens": max_tokens,
            });

            let res = client
                .post("https://integrate.api.nvidia.com/v1/chat/completions")
                .header("Authorization", format!("Bearer {}", api_key.trim()))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await;

            match res {
                Ok(resp) => {
                    if resp.status().is_success() {
                        if let Ok(json_body) = resp.json::<serde_json::Value>().await {
                            return Ok(Json(json_body).into_response());
                        }
                    } else {
                        let err_text = resp.text().await.unwrap_or_default();
                        eprintln!("Error de upstream Nemotron: {}", err_text);
                    }
                }
                Err(e) => {
                    eprintln!("Fallo de red conectando con upstream Nemotron: {}", e);
                }
            }
        }
    }

    // 2. Fallback: Construir prompt a partir de messages para el motor local continuo
    let mut prompt = String::new();
    for msg in &messages {
        prompt.push_str(&format!("{}: {}\n", msg.role, msg.content));
    }
    prompt.push_str("assistant: ");

    let prompt_tokens = state
        .scheduler
        .tokenizer
        .encode(&prompt)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Error de tokenización: {}", e)))?;

    if prompt_tokens.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Prompt no puede estar vacío".to_string()));
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

    // 2. Si el cliente solicitó Streaming vía SSE:
    if payload.stream {
        let sse = create_sse_response(seq_id, state.config.model_id.clone(), rx);
        return Ok(sse.into_response());
    }

    // 3. Si solicitó respuesta síncrona completa:
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
