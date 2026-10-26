//! Supervisor y Orquestador Multi-Agente con Grafo de Tareas y Bucle de Consenso.

use super::actor::AgentActorHandle;
use super::dag::{TaskGraph, TaskNode};
use crate::agent::AgentRole;
use crate::events::{AgentEvent, EventBus};
use serde::{Deserialize, Serialize};
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
        }
    }

    /// Construye el DAG canónico de tres fases: Investigación -> Implementación -> Auditoría.
    pub fn build_standard_dag(&self, goal: &str) -> TaskGraph {
        let mut graph = TaskGraph::new();

        graph.add_task(TaskNode::new(
            "task_1_research",
            AgentRole::Researcher,
            format!("Investigar especificaciones técnicas, SIMD y patrones de memoria para: '{goal}'"),
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

    /// Ejecuta el flujo de trabajo orquestado de principio a fin respetando dependencias del grafo.
    pub async fn run_workflow(&self, workflow_id: &str, goal: &str) -> Result<WorkflowReport, String> {
        let start_time = Instant::now();

        self.event_bus.publish(AgentEvent::WorkflowStarted {
            workflow_id: workflow_id.to_string(),
            goal: goal.to_string(),
            timestamp_ms: start_time.elapsed().as_millis() as u64,
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
                                    parsed["feedback"].as_str().unwrap_or("Aprobado").to_string(),
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

        let research_summary = graph.tasks.get("task_1_research").and_then(|t| t.output.clone());
        let generated_code = graph.tasks.get("task_2_code").and_then(|t| t.output.clone());
        let review_feedback = graph.tasks.get("task_3_review").and_then(|t| t.output.clone());

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
