//! Motor de ejecución del bucle ReAct (Reasoning + Acting).

use super::llm_client::{LlmBackend, LlmDecision};
use super::state::{AgentContext, AgentMessage, AgentState, ReActStep};
use crate::events::{AgentEvent, EventBus};
use crate::tools::ToolRegistry;
use std::sync::Arc;
use std::time::Instant;

/// Resultado de la ejecución de una tarea por un agente en su bucle ReAct.
#[derive(Debug, Clone)]
pub struct ReActResult {
    pub final_answer: String,
    pub total_steps: usize,
    pub state: AgentState,
}

/// Motor de ejecución del ciclo ReAct con observabilidad por eventos y límites de seguridad.
pub struct ReActEngine {
    llm: Arc<dyn LlmBackend>,
    tools: ToolRegistry,
    event_bus: Option<EventBus>,
}

impl ReActEngine {
    /// Inicializa un nuevo motor ReAct.
    pub fn new(llm: Arc<dyn LlmBackend>, tools: ToolRegistry) -> Self {
        Self {
            llm,
            tools,
            event_bus: None,
        }
    }

    /// Adjunta un bus de eventos para observabilidad y streaming en tiempo real.
    pub fn with_event_bus(mut self, bus: EventBus) -> Self {
        self.event_bus = Some(bus);
        self
    }

    /// Ejecuta el bucle ReAct hasta que el agente llega a una respuesta final o agota las iteraciones.
    pub async fn run(&self, context: &mut AgentContext) -> Result<ReActResult, String> {
        let mut iteration = 0;

        while iteration < context.max_iterations {
            iteration += 1;

            // 1. Fase THOUGHT (Razonamiento)
            let decision = self
                .llm
                .decide(context, &self.tools)
                .await
                .map_err(|e| format!("LLM reasoning failure: {e}"))?;

            match decision {
                LlmDecision::ToolCall {
                    thought,
                    tool_name,
                    arguments,
                } => {
                    // Notificar pensamiento al bus de eventos
                    if let Some(bus) = &self.event_bus {
                        bus.publish(AgentEvent::AgentThinking {
                            agent_role: context.role.to_string(),
                            thought: thought.clone(),
                        });
                        bus.publish(AgentEvent::ToolCallStarted {
                            agent_role: context.role.to_string(),
                            tool_name: tool_name.clone(),
                            arguments: arguments.clone(),
                        });
                    }

                    // 2. Fase ACTION (Ejecución de herramienta)
                    let start_time = Instant::now();
                    let exec_result = self.tools.execute(&tool_name, arguments.clone()).await;
                    let duration_ms = start_time.elapsed().as_millis() as u64;

                    let (success, output_val) = match exec_result {
                        Ok(out) => (out.success, out.result),
                        Err(err) => (false, serde_json::json!({ "error": err.to_string() })),
                    };

                    // 3. Fase OBSERVATION (Observación del resultado)
                    if let Some(bus) = &self.event_bus {
                        bus.publish(AgentEvent::ToolCallCompleted {
                            agent_role: context.role.to_string(),
                            tool_name: tool_name.clone(),
                            success,
                            output: output_val.clone(),
                            duration_ms,
                        });
                    }

                    let step = ReActStep {
                        iteration,
                        thought,
                        action: Some((tool_name.clone(), arguments)),
                        observation: Some(output_val.clone()),
                        reflection: Some(format!("Herramienta '{tool_name}' completada en {duration_ms}ms")),
                    };

                    context.record_step(step);
                    context.add_message(AgentMessage::tool(tool_name, output_val.to_string()));
                }

                LlmDecision::FinalAnswer { thought, answer } => {
                    // Notificar conclusión al bus
                    if let Some(bus) = &self.event_bus {
                        bus.publish(AgentEvent::AgentThinking {
                            agent_role: context.role.to_string(),
                            thought: thought.clone(),
                        });
                    }

                    let step = ReActStep {
                        iteration,
                        thought,
                        action: None,
                        observation: None,
                        reflection: Some("Objetivo completado satisfactoriamente.".into()),
                    };

                    context.record_step(step);
                    context.add_message(AgentMessage::assistant(answer.clone()));

                    return Ok(ReActResult {
                        final_answer: answer,
                        total_steps: context.steps.len(),
                        state: AgentState::Completed,
                    });
                }
            }
        }

        // Límite de iteraciones alcanzado
        let fail_reason = format!("Max iterations limit ({}) exceeded", context.max_iterations);
        Err(fail_reason)
    }
}
