//! Cliente asíncrono para NVIDIA Nemotron con soporte de streaming SSE y modo simulador.

use std::time::Duration;
use futures_util::StreamExt;
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use tokio::sync::mpsc;

use crate::config::{BackendMode, NemotronConfig};
use crate::sse::{SseEvent, SseParser};
use crate::types::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage};

pub struct NemotronClient {
    pub config: NemotronConfig,
    http_client: reqwest::Client,
}

impl NemotronClient {
    pub fn new(config: NemotronConfig) -> Result<Self, String> {
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| format!("Error inicializando reqwest Client: {e}"))?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Envío de mensaje en modo síncrono/bloqueante completo (sin streaming)
    pub async fn complete(&self, messages: &[ChatMessage]) -> Result<ChatCompletionResponse, String> {
        match self.config.backend_mode {
            BackendMode::MockSimulator => {
                tokio::time::sleep(Duration::from_millis(300)).await;
                let last_user_msg = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::types::Role::User)
                    .map(|m| m.content.as_str())
                    .unwrap_or("hola");

                let mock_text = generate_mock_nemotron_response(last_user_msg);
                Ok(ChatCompletionResponse {
                    id: "mock-nemotron-70b-001".to_string(),
                    model: self.config.model.clone(),
                    choices: vec![crate::types::ChatChoice {
                        index: 0,
                        message: ChatMessage::assistant(mock_text),
                        finish_reason: Some("stop".to_string()),
                    }],
                    usage: Some(crate::types::Usage {
                        prompt_tokens: 42,
                        completion_tokens: 128,
                        total_tokens: 170,
                    }),
                })
            }
            BackendMode::NvidiaNim | BackendMode::LocalEngine => {
                let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                if let Some(key) = &self.config.api_key {
                    let auth_val = format!("Bearer {key}");
                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&auth_val).map_err(|e| format!("Header inválido: {e}"))?,
                    );
                }

                let req_body = ChatCompletionRequest {
                    model: self.config.model.clone(),
                    messages: messages.to_vec(),
                    temperature: self.config.temperature,
                    top_p: self.config.top_p,
                    max_tokens: self.config.max_tokens,
                    stream: false,
                };

                let response = self
                    .http_client
                    .post(&url)
                    .headers(headers)
                    .json(&req_body)
                    .send()
                    .await
                    .map_err(|e| format!("Fallo de conexión HTTP con {url}: {e}"))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let err_text = response.text().await.unwrap_or_default();
                    return Err(format!("Error de API Nemotron ({status}): {err_text}"));
                }

                response
                    .json::<ChatCompletionResponse>()
                    .await
                    .map_err(|e| format!("Fallo deserializando respuesta JSON: {e}"))
            }
        }
    }

    /// Envío de mensaje en modo Streaming continuo token por token mediante un canal `mpsc`
    pub async fn stream_chat(
        &self,
        messages: &[ChatMessage],
    ) -> Result<mpsc::Receiver<Result<String, String>>, String> {
        let (tx, rx) = mpsc::channel(64);

        match self.config.backend_mode {
            BackendMode::MockSimulator => {
                let last_user_msg = messages
                    .iter()
                    .rev()
                    .find(|m| m.role == crate::types::Role::User)
                    .map(|m| m.content.to_string())
                    .unwrap_or_else(|| "hola".to_string());

                tokio::spawn(async move {
                    let mock_text = generate_mock_nemotron_response(&last_user_msg);
                    // Simular TTFT de red neuronal
                    tokio::time::sleep(Duration::from_millis(180)).await;

                    // Emitir tokens con ritmo de máquina de escribir fluida (50-60 TPS)
                    for word in mock_text.split_inclusive(' ') {
                        if tx.send(Ok(word.to_string())).await.is_err() {
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(18)).await;
                    }
                });

                Ok(rx)
            }
            BackendMode::NvidiaNim | BackendMode::LocalEngine => {
                let url = format!("{}/chat/completions", self.config.base_url.trim_end_matches('/'));
                let mut headers = HeaderMap::new();
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
                if let Some(key) = &self.config.api_key {
                    let auth_val = format!("Bearer {key}");
                    headers.insert(
                        AUTHORIZATION,
                        HeaderValue::from_str(&auth_val).map_err(|e| format!("Header inválido: {e}"))?,
                    );
                }

                let req_body = ChatCompletionRequest {
                    model: self.config.model.clone(),
                    messages: messages.to_vec(),
                    temperature: self.config.temperature,
                    top_p: self.config.top_p,
                    max_tokens: self.config.max_tokens,
                    stream: true,
                };

                let response = self
                    .http_client
                    .post(&url)
                    .headers(headers)
                    .json(&req_body)
                    .send()
                    .await
                    .map_err(|e| format!("Fallo conectando a {url}: {e}"))?;

                if !response.status().is_success() {
                    let status = response.status();
                    let err_body = response.text().await.unwrap_or_default();
                    return Err(format!("Error en stream Nemotron ({status}): {err_body}"));
                }

                let mut stream = response.bytes_stream();

                tokio::spawn(async move {
                    let mut buffer = String::new();

                    while let Some(chunk_res) = stream.next().await {
                        match chunk_res {
                            Ok(bytes) => {
                                if let Ok(text) = std::str::from_utf8(&bytes) {
                                    buffer.push_str(text);

                                    // Procesar líneas delimitadas por \n
                                    while let Some(pos) = buffer.find('\n') {
                                        let line = buffer[..pos].to_string();
                                        buffer = buffer[pos + 1..].to_string();

                                        match SseParser::parse_line(&line) {
                                            Ok(SseEvent::Token(token)) => {
                                                if tx.send(Ok(token)).await.is_err() {
                                                    return;
                                                }
                                            }
                                            Ok(SseEvent::Done) => return,
                                            Ok(SseEvent::Comment(_)) | Ok(SseEvent::Empty) => {}
                                            Err(e) => {
                                                let _ = tx.send(Err(e)).await;
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(format!("Error de lectura en byte stream: {e}"))).await;
                                return;
                            }
                        }
                    }
                });

                Ok(rx)
            }
        }
    }
}

/// Generador de respuestas técnicas simuladas para el modo de prueba offline
fn generate_mock_nemotron_response(user_input: &str) -> String {
    let lower = user_input.to_lowercase();
    if lower.contains("rust") || lower.contains("codigo") || lower.contains("código") {
        "Como modelo Nemotron de NVIDIA, optimizo código para máximo rendimiento en silicio moderno. \
        En Rust, la clave para procesar tokens a alta velocidad radica en evitar copias innecesarias en el heap: \
        utiliza slices `&[T]`, buffers contiguos con pre-asignación `Vec::with_capacity` y canales asíncronos `tokio::sync::mpsc`. \
        ¿Deseas que examinemos un ejemplo de deserialización zero-copy o la integración con CUDA bindings?".to_string()
    } else if lower.contains("hardware") || lower.contains("gpu") || lower.contains("nvidia") {
        "En arquitecturas de aceleración NVIDIA (Ampere, Hopper y Blackwell), los modelos Nemotron \
        aprovechan los Tensor Cores de 4ta y 5ta generación con precisión reducida FP8 y NVFP4. \
        Al sincronizar streams con Rust y C FFI, se logra eliminar la sobrecarga del runtime de Python, \
        reduciendo la latencia de despacho de kernels a menos de 5 microsegundos.".to_string()
    } else {
        format!(
            "Hola, soy NVIDIA Nemotron. He recibido tu mensaje: \"{}\". \
            Estoy listo para ayudarte con diseño de sistemas en Rust, optimización de algoritmos de deep learning, \
            análisis de arquitecturas de transformers o ingeniería de datos de alto rendimiento. ¿Por dónde empezamos?",
            user_input.trim()
        )
    }
}
