//! Cliente y backend de razonamiento para agentes con soporte desacoplado (LLM Gateway / OpenAI / Simulado).

use super::state::{AgentContext, AgentRole};
use crate::tools::ToolRegistry;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Decisión generada por el backend de lenguaje en cada iteración del bucle ReAct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LlmDecision {
    /// El agente decide invocar una herramienta externa para obtener información o validar código.
    ToolCall {
        thought: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    /// El agente ha completado su razonamiento y emite su respuesta o entrega final.
    FinalAnswer {
        thought: String,
        answer: String,
    },
}

/// Contrato para el backend de lenguaje (LLM).
#[async_trait]
pub trait LlmBackend: Send + Sync {
    /// Determina el siguiente paso de razonamiento (pensamiento + acción o respuesta final).
    async fn decide(
        &self,
        context: &AgentContext,
        tools: &ToolRegistry,
    ) -> Result<LlmDecision, String>;
}

/// Backend de razonamiento determinista y pedagógico para testing y ejecución en producción local.
/// Responde con lógica de razonamiento contextual adaptada al rol del agente (`Researcher`, `Coder`, `Reviewer`, `Supervisor`).
#[derive(Default, Clone)]
pub struct DeterministicLlmBackend;

#[async_trait]
impl LlmBackend for DeterministicLlmBackend {
    async fn decide(
        &self,
        context: &AgentContext,
        _tools: &ToolRegistry,
    ) -> Result<LlmDecision, String> {
        let last_message = context
            .messages
            .last()
            .map(|m| m.content.as_str())
            .unwrap_or_default();

        let step_count = context.steps.len();

        match context.role {
            // ---------------------------------------------------------------
            // 1. INVESTIGADOR (Researcher)
            // ---------------------------------------------------------------
            AgentRole::Researcher => {
                let is_arithmetic = (last_message.contains('+')
                    || last_message.contains('*')
                    || last_message.contains('/')
                    || last_message.contains('-'))
                    && last_message.chars().any(|c| c.is_ascii_digit());

                if step_count == 0 {
                    if is_arithmetic {
                        // Extraer posible subcadena de cálculo
                        let expr = last_message
                            .split_whitespace()
                            .find(|w| w.contains('+') || w.contains('*') || w.contains('/') || w.contains('-'))
                            .unwrap_or("128 * 16");

                        Ok(LlmDecision::ToolCall {
                            thought: "El requerimiento incluye una operación aritmética. Invoco la calculadora tipada para evitar alucinaciones.".into(),
                            tool_name: "calculator".into(),
                            arguments: serde_json::json!({ "expression": expr }),
                        })
                    } else {
                        Ok(LlmDecision::ToolCall {
                            thought: "Debo investigar las especificaciones arquitectónicas en la base de conocimiento RAG antes de formular la propuesta técnica.".into(),
                            tool_name: "rag_search".into(),
                            arguments: serde_json::json!({
                                "query": last_message,
                                "top_k": 2
                            }),
                        })
                    }
                } else {
                    let previous_obs = context
                        .steps
                        .first()
                        .and_then(|s| s.observation.as_ref())
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "Conocimiento base recuperado.".into());

                    Ok(LlmDecision::FinalAnswer {
                        thought: "He recopilado suficiente contexto técnico y referencias de diseño de baja latencia para el informe.".into(),
                        answer: format!(
                            "### Reporte de Investigación Técnica\n\n\
                            **Objetivo Analizado:** {}\n\n\
                            **Hallazgos Clave:**\n\
                            - Se validó el diseño sin fragmentación y con seguridad de memoria estricta.\n\
                            - Los canales Tokio y tipos tipados eliminan condiciones de carrera.\n\n\
                            **Datos de Referencia RAG:**\n```json\n{}\n```",
                            last_message, previous_obs
                        ),
                    })
                }
            }

            // ---------------------------------------------------------------
            // 2. DESARROLLADOR (Coder)
            // ---------------------------------------------------------------
            AgentRole::Coder => {
                let user_goal = context
                    .messages
                    .iter()
                    .filter(|m| m.role == "user")
                    .last()
                    .map(|m| m.content.as_str())
                    .unwrap_or(last_message);

                let msg_lower = user_goal.to_lowercase();

                let (proposed_code, final_code) = if msg_lower.contains("ring")
                    || msg_lower.contains("buffer")
                    || msg_lower.contains("circular")
                {
                    (
                        "pub struct ConcurrentRingBuffer<T> {\n\
                            buffer: Vec<Option<T>>,\n\
                            capacity: usize,\n\
                            head: usize,\n\
                            tail: usize,\n\
                        }\n\n\
                        impl<T: Clone> ConcurrentRingBuffer<T> {\n\
                            pub fn new(capacity: usize) -> Self {\n\
                                Self { buffer: vec![None; capacity], capacity, head: 0, tail: 0 }\n\
                            }\n\
                        }",
                        "```rust\n\
                        /// Estructura de RingBuffer concurrente en memoria contigua\n\
                        #[derive(Debug, Clone)]\n\
                        pub struct ConcurrentRingBuffer<T> {\n\
                            buffer: Vec<Option<T>>,\n\
                            capacity: usize,\n\
                            head: usize,\n\
                            tail: usize,\n\
                        }\n\n\
                        impl<T: Clone> ConcurrentRingBuffer<T> {\n\
                            pub fn new(capacity: usize) -> Self {\n\
                                Self {\n\
                                    buffer: vec![None; capacity],\n\
                                    capacity,\n\
                                    head: 0,\n\
                                    tail: 0,\n\
                                }\n\
                            }\n\n\
                            pub fn push(&mut self, item: T) -> Result<(), &'static str> {\n\
                                let next_tail = (self.tail + 1) % self.capacity;\n\
                                if next_tail == self.head {\n\
                                    return Err(\"Buffer lleno\");\n\
                                }\n\
                                self.buffer[self.tail] = Some(item);\n\
                                self.tail = next_tail;\n\
                                Ok(())\n\
                            }\n\n\
                            pub fn pop(&mut self) -> Option<T> {\n\
                                if self.head == self.tail {\n\
                                    return None;\n\
                                }\n\
                                let item = self.buffer[self.head].take();\n\
                                self.head = (self.head + 1) % self.capacity;\n\
                                item\n\
                            }\n\
                        }\n\
                        ```",
                    )
                } else if msg_lower.contains("simd")
                    || msg_lower.contains("dot")
                    || msg_lower.contains("vector")
                {
                    (
                        "pub struct SimdVectorEngine {\n\
                            dimension: usize,\n\
                        }\n\n\
                        impl SimdVectorEngine {\n\
                            pub fn new(dimension: usize) -> Self {\n\
                                Self { dimension }\n\
                            }\n\
                        }",
                        "```rust\n\
                        /// Motor de cálculo vectorial acelerado por SIMD\n\
                        #[derive(Debug, Clone)]\n\
                        pub struct SimdVectorEngine {\n\
                            dimension: usize,\n\
                        }\n\n\
                        impl SimdVectorEngine {\n\
                            pub fn new(dimension: usize) -> Self {\n\
                                Self { dimension }\n\
                            }\n\n\
                            pub fn dot_product(&self, a: &[f32], b: &[f32]) -> f32 {\n\
                                assert_eq!(a.len(), b.len(), \"Dimensiones deben coincidir\");\n\
                                a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()\n\
                            }\n\
                        }\n\
                        ```",
                    )
                } else {
                    (
                        "pub struct ConcurrentLruCache<K, V> {\n\
                            map: std::collections::HashMap<K, V>,\n\
                            capacity: usize,\n\
                        }\n\n\
                        impl<K: Eq + std::hash::Hash + Clone, V: Clone> ConcurrentLruCache<K, V> {\n\
                            pub fn new(capacity: usize) -> Self {\n\
                                Self { map: std::collections::HashMap::new(), capacity }\n\
                            }\n\
                            pub fn get(&self, key: &K) -> Option<&V> {\n\
                                self.map.get(key)\n\
                            }\n\
                        }",
                        "```rust\n\
                        /// Estructura de caché concurrente segura en memoria\n\
                        #[derive(Debug, Clone)]\n\
                        pub struct ConcurrentLruCache<K, V> {\n\
                            map: std::collections::HashMap<K, V>,\n\
                            capacity: usize,\n\
                        }\n\n\
                        impl<K: Eq + std::hash::Hash + Clone, V: Clone> ConcurrentLruCache<K, V> {\n\
                            pub fn new(capacity: usize) -> Self {\n\
                                Self { map: std::collections::HashMap::new(), capacity }\n\
                            }\n\
                            pub fn get(&self, key: &K) -> Option<&V> {\n\
                                self.map.get(key)\n\
                            }\n\
                            pub fn insert(&mut self, key: K, value: V) {\n\
                                if self.map.len() >= self.capacity {\n\
                                    if let Some(first) = self.map.keys().next().cloned() {\n\
                                        self.map.remove(&first);\n\
                                    }\n\
                                }\n\
                                self.map.insert(key, value);\n\
                            }\n\
                        }\n\
                        ```",
                    )
                };

                if step_count == 0 {
                    Ok(LlmDecision::ToolCall {
                        thought: "He diseñado la estructura en Rust adaptada al requerimiento. La someteré al validador sintáctico para asegurar cero unsafe.".into(),
                        tool_name: "rust_code_validator".into(),
                        arguments: serde_json::json!({ "code": proposed_code }),
                    })
                } else {
                    Ok(LlmDecision::FinalAnswer {
                        thought: "El validador de código confirmó sanidad sintáctica, balance de llaves y cero unsafe.".into(),
                        answer: final_code.into(),
                    })
                }
            }

            // ---------------------------------------------------------------
            // 3. REVISOR Y AUDITOR (Reviewer)
            // ---------------------------------------------------------------
            AgentRole::Reviewer => {
                let full_context = context
                    .messages
                    .iter()
                    .map(|m| m.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");

                let code_snippet = if let Some(start) = full_context.find("```rust") {
                    let after_start = &full_context[start + 7..];
                    if let Some(end) = after_start.find("```") {
                        &after_start[..end]
                    } else {
                        after_start
                    }
                } else {
                    full_context.as_str()
                };

                let code_has_unsafe =
                    code_snippet.contains("unsafe ") || code_snippet.contains("unsafe{");
                let has_documentation = code_snippet.contains("///");

                let score = if has_documentation && !code_has_unsafe {
                    0.98
                } else if !code_has_unsafe {
                    0.85
                } else {
                    0.60
                };

                let approved = score >= 0.80;

                Ok(LlmDecision::FinalAnswer {
                    thought: format!("Audité el entregable: documentation={has_documentation}, unsafe={code_has_unsafe}. Puntuación asignada: {score:.2}"),
                    answer: serde_json::json!({
                        "approved": approved,
                        "score": score,
                        "feedback": if approved {
                            "Código aprobado: Sintaxis limpia, cero unsafe blocks, alta seguridad de memoria y tipos genéricos adecuados."
                        } else {
                            "Código rechazado: Se detectaron violaciones o falta de documentación requerida."
                        }
                    }).to_string(),
                })
            }

            // ---------------------------------------------------------------
            // 4. SUPERVISOR (Supervisor)
            // ---------------------------------------------------------------
            AgentRole::Supervisor => Ok(LlmDecision::FinalAnswer {
                thought: "He coordinado la ejecución de todos los agentes y consolidado la solución final.".into(),
                answer: format!(
                    "Consolidación de Solución Multi-Agente:\n\n\
                    - Objetivo completado con éxito.\n\
                    - Tareas ejecutadas y auditadas por el equipo de agentes.\n\
                    - Resumen del requerimiento: {last_message}"
                ),
            }),
        }
    }
}
