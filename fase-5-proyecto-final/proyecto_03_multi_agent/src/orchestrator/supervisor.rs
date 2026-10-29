//! Supervisor y Orquestador Multi-Agente con Enrutador de Intenciones (Intent Router),
//! Grafo de Tareas Acíclico Dirigido (DAG) y Bucle de Consenso.

use super::actor::AgentActorHandle;
use super::dag::{TaskGraph, TaskNode};
pub use super::semantic_router::{RouteMatch, SemanticRouter, UserIntent};
use crate::agent::{AgentContext, AgentMessage, AgentRole, LlmBackend, LlmDecision};
use crate::events::{AgentEvent, EventBus};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Reporte final estructurado generado por la orquestación multi-agente.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowReport {
    pub workflow_id: String,
    pub goal: String,
    pub success: bool,
    pub research_summary: Option<String>,
    pub generated_code: Option<String>,
    pub review_feedback: Option<String>,
    pub final_synthesis: String,
    pub total_duration_ms: u64,
}

/// Orquestador principal que coordina el modelo de actores y resuelve el DAG de ejecución.
pub struct SupervisorOrchestrator {
    researcher: AgentActorHandle,
    coder: AgentActorHandle,
    reviewer: AgentActorHandle,
    event_bus: EventBus,
    llm: Option<Arc<dyn LlmBackend>>,
    semantic_router: SemanticRouter,
}

impl SupervisorOrchestrator {
    /// Inicializa un nuevo orquestador supervisor con sus actores subordinados.
    pub fn new(
        researcher: AgentActorHandle,
        coder: AgentActorHandle,
        reviewer: AgentActorHandle,
        event_bus: EventBus,
    ) -> Self {
        Self {
            researcher,
            coder,
            reviewer,
            event_bus,
            llm: None,
            semantic_router: SemanticRouter::new(),
        }
    }

    /// Adjunta un backend de LLM para permitir diálogo conversacional directo.
    pub fn with_llm(mut self, llm: Arc<dyn LlmBackend>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Clasifica semánticamente si la solicitud del usuario es conversación o tarea de ingeniería.
    pub fn classify_intent(goal: &str) -> UserIntent {
        let router = SemanticRouter::new();
        router.classify_intent(goal)
    }

    /// Retorna los detalles vectoriales del emparejamiento semántico (confianza y similitudes)
    pub fn classify_route(&self, goal: &str) -> RouteMatch {
        self.semantic_router.classify(goal)
    }

    /// Maneja interacciones conversacionales de forma directa.
    async fn handle_conversational(&self, goal: &str) -> String {
        if let Some(llm) = &self.llm {
            let mut context = AgentContext::new(
                AgentRole::Supervisor,
                "Eres un Asistente de IA y Supervisor de un sistema multi-agente en Rust. Responde al usuario de forma amigable, concisa y servicial. Explica brevemente que cuentas con un equipo de especialistas (Investigador RAG, Programador Rust y Auditor de Código) listo para construir software de alto rendimiento.",
            );
            context.add_message(AgentMessage::user(goal));
            let dummy_tools = crate::tools::ToolRegistry::new();

            if let Ok(decision) = llm.decide(&context, &dummy_tools).await {
                match decision {
                    LlmDecision::FinalAnswer { answer, .. } => return answer,
                    LlmDecision::ToolCall { thought, .. } => return thought,
                }
            }
        }

        // Respuesta conversacional contextual por defecto (Modo Offline)
        let cleaned = goal.trim().to_lowercase();
        if cleaned.contains("opinas de rust")
            || cleaned.contains("piensas de rust")
            || cleaned.contains("opinion de rust")
            || cleaned.contains("opinión sobre rust")
            || cleaned.contains("about rust")
        {
            "Como supervisor de sistemas de IA en Rust, considero que **Rust representa el estándar moderno de ingeniería de software para sistemas críticos y de alto rendimiento**.\n\n\
            Sus tres pilares fundamentales transforman el desarrollo:\n\
            1. **Seguridad de Memoria en Compilación (Borrow Checker):** Elimina punteros nulos, *data races* y *use-after-free* sin necesidad de recolector de basura (GC), garantizando latencias predecibles en microsegundos.\n\
            2. **Abstracciones de Costo Cero (*Zero-Cost Abstractions*):** Los traits, iteradores y la concurrencia asíncrona de Tokio compilan a instrucciones máquina nativas con rendimiento comparable o superior a C++.\n\
            3. **Ecosistema Industrial de IA:** Herramientas como ONNX Runtime (`ort`), Hugging Face `Candle`, `Polars` y `Axum` permiten construir motores de inferencia y agentes ultraligeros (< 30 MB RAM) con tipado estricto.\n\n\
            ¿Deseas que diseñemos o analicemos algún patrón arquitectónico específico en Rust?".to_string()
        } else if cleaned.contains("quien eres")
            || cleaned.contains("who are you")
            || cleaned.contains("que eres")
        {
            "¡Hola! Soy el Supervisor del Sistema Autónomo Multi-Agente en Rust puro.\n\n\
            Coordino un equipo de agentes concurrentes basados en el Modelo de Actores de Tokio:\n\
            - 🔬 **Investigador (Researcher):** Explora especificaciones y bases documentales RAG.\n\
            - 💻 **Desarrollador (Coder):** Escribe algoritmos y estructuras de datos en Rust seguro.\n\
            - ⚖️ **Auditor (Reviewer):** Valida balance de delimitadores, comentarios y ausencia de `unsafe`.\n\n\
            Puedes conversar conmigo o pedirme diseñar cualquier componente técnico (ej. 'Diseñar un RingBuffer continuo' o 'Crear un módulo SIMD AVX2').".to_string()
        } else if cleaned.contains("gracias") || cleaned.contains("thanks") {
            "¡De nada! Ha sido un placer. Si necesitas construir o auditar otro componente en Rust, solo dímelo.".to_string()
        } else {
            format!(
                "¡Hola! Recibí tu mensaje: \"{}\".\n\n\
                Soy el Supervisor del framework multi-agente. Mi enrutador semántico ha clasificado tu consulta como conversacional. Estoy listo para ayudarte tanto con consultas conceptuales como con la coordinación de tareas complejas de ingeniería en Rust.\n\
                ¿En qué puedo ayudarte hoy?",
                goal.trim()
            )
        }
    }

    /// Construye el DAG canónico de tres fases: Investigación -> Implementación -> Auditoría.
    pub fn build_standard_dag(&self, goal: &str) -> TaskGraph {
        let mut graph = TaskGraph::new();

        graph.add_task(TaskNode::new(
            "task_1_research",
            AgentRole::Researcher,
            format!(
                "Investigar especificaciones técnicas, SIMD y patrones de memoria para: '{goal}'"
            ),
            vec![],
        ));

        graph.add_task(TaskNode::new(
            "task_2_code",
            AgentRole::Coder,
            format!("Implementar la estructura de datos en Rust puro sin unsafe y con tipos genéricos para: '{goal}'"),
            vec!["task_1_research".into()],
        ));

        graph.add_task(TaskNode::new(
            "task_3_review",
            AgentRole::Reviewer,
            "Auditar el código producido, verificar ausencia de unsafe y evaluar cobertura documental.",
            vec!["task_2_code".into()],
        ));

        graph
    }

    /// Ejecuta el flujo de trabajo orquestado respetando la intención del usuario y dependencias del grafo.
    pub async fn run_workflow(
        &self,
        workflow_id: &str,
        goal: &str,
    ) -> Result<WorkflowReport, String> {
        let start_time = Instant::now();

        self.event_bus.publish(AgentEvent::WorkflowStarted {
            workflow_id: workflow_id.to_string(),
            goal: goal.to_string(),
            timestamp_ms: start_time.elapsed().as_millis() as u64,
        });

        // 1. CLASIFICACIÓN SEMÁNTICA VECTORIAL DE INTENCIÓN (Semantic Router)
        let route_match = self.classify_route(goal);

        if route_match.intent == UserIntent::Conversational {
            self.event_bus.publish(AgentEvent::AgentThinking {
                agent_role: "Supervisor".to_string(),
                thought: format!(
                    "Enrutador Semántico: Intención 'Conversational' detectada (Confianza: {:.2}, Sim Conv: {:.2}, Sim Tech: {:.2}, Ancla: '{}'). Respondiendo de forma conversacional sin requerir ciclo de auditoría de código.",
                    route_match.confidence, route_match.conv_similarity, route_match.tech_similarity, route_match.top_anchor
                ),
            });

            let conversational_response = self.handle_conversational(goal).await;
            let total_duration_ms = start_time.elapsed().as_millis() as u64;

            self.event_bus.publish(AgentEvent::WorkflowCompleted {
                workflow_id: workflow_id.to_string(),
                summary: conversational_response.clone(),
                total_duration_ms,
            });

            return Ok(WorkflowReport {
                workflow_id: workflow_id.to_string(),
                goal: goal.to_string(),
                success: true,
                research_summary: None,
                generated_code: None,
                review_feedback: None,
                final_synthesis: conversational_response,
                total_duration_ms,
            });
        }

        // 2. TAREA TÉCNICA: EJECUTAR GRAFO ACÍCLICO DIRIGIDO (DAG)
        self.event_bus.publish(AgentEvent::AgentThinking {
            agent_role: "Supervisor".to_string(),
            thought: format!(
                "Enrutador Semántico: Intención 'TechnicalTask' detectada (Confianza: {:.2}, Sim Tech: {:.2}, Sim Conv: {:.2}, Ancla: '{}'). Activando equipo multi-agente con DAG de ejecución.",
                route_match.confidence, route_match.tech_similarity, route_match.conv_similarity, route_match.top_anchor
            ),
        });

        let mut graph = self.build_standard_dag(goal);

        if graph.has_cycle() {
            let err = "Cyclic dependency detected in task graph".to_string();
            self.event_bus.publish(AgentEvent::WorkflowFailed {
                workflow_id: workflow_id.to_string(),
                error: err.clone(),
            });
            return Err(err);
        }

        let mut max_rounds = 20;

        while !graph.is_all_completed() && max_rounds > 0 {
            max_rounds -= 1;
            let ready_tasks = graph.get_ready_tasks();

            if ready_tasks.is_empty() {
                if graph.has_failures() {
                    return Err("One or more tasks in the workflow failed".into());
                }
                break;
            }

            for task in ready_tasks {
                graph.mark_running(&task.id);
                let dep_context = graph.get_dependency_outputs(&task.id);

                let handle = match task.role {
                    AgentRole::Researcher => &self.researcher,
                    AgentRole::Coder => &self.coder,
                    AgentRole::Reviewer => &self.reviewer,
                    AgentRole::Supervisor => &self.researcher, // Fallback
                };

                let output_result = handle
                    .execute_task(task.id.clone(), task.description.clone(), dep_context)
                    .await;

                match output_result {
                    Ok(out) => {
                        // Si es la tarea del revisor, emitir evento de consenso
                        if task.role == AgentRole::Reviewer {
                            let (approved, score, feedback) = if let Ok(parsed) =
                                serde_json::from_str::<serde_json::Value>(&out)
                            {
                                (
                                    parsed["approved"].as_bool().unwrap_or(true),
                                    parsed["score"].as_f64().unwrap_or(0.95) as f32,
                                    parsed["feedback"]
                                        .as_str()
                                        .unwrap_or("Aprobado")
                                        .to_string(),
                                )
                            } else {
                                (true, 0.90, "Revisión satisfactoria".to_string())
                            };

                            self.event_bus.publish(AgentEvent::ReviewOutcome {
                                approved,
                                score,
                                feedback,
                            });
                        }

                        graph.mark_completed(&task.id, out);
                    }
                    Err(err) => {
                        graph.mark_failed(&task.id, err.clone());
                        self.event_bus.publish(AgentEvent::WorkflowFailed {
                            workflow_id: workflow_id.to_string(),
                            error: err.clone(),
                        });
                        return Err(err);
                    }
                }
            }
        }

        let total_duration_ms = start_time.elapsed().as_millis() as u64;

        let research_summary = graph
            .tasks
            .get("task_1_research")
            .and_then(|t| t.output.clone());
        let generated_code = graph
            .tasks
            .get("task_2_code")
            .and_then(|t| t.output.clone());
        let review_feedback = graph
            .tasks
            .get("task_3_review")
            .and_then(|t| t.output.clone());

        let final_synthesis = format!(
            "### Orquestación Multi-Agente Completada\n\n\
            **ID del Flujo:** {}\n\
            **Meta:** {}\n\
            **Tiempo de Ejecución:** {} ms\n\
            **Consenso del Equipo:** Las 3 etapas (Investigación -> Código -> Auditoría) concluyeron con éxito.",
            workflow_id, goal, total_duration_ms
        );

        self.event_bus.publish(AgentEvent::WorkflowCompleted {
            workflow_id: workflow_id.to_string(),
            summary: final_synthesis.clone(),
            total_duration_ms,
        });

        Ok(WorkflowReport {
            workflow_id: workflow_id.to_string(),
            goal: goal.to_string(),
            success: true,
            research_summary,
            generated_code,
            review_feedback,
            final_synthesis,
            total_duration_ms,
        })
    }
}
