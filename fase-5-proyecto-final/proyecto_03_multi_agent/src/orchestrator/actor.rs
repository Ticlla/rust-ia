//! Implementación del Modelo de Actores para agentes mediante canales Tokio (`mpsc` + `oneshot`).

use crate::agent::{AgentContext, AgentRole, AgentState, ReActEngine};
use crate::events::{AgentEvent, EventBus};
use std::sync::{Arc, RwLock};
use tokio::sync::{mpsc, oneshot};

/// Mensajes admitidos por el buzón de entrada de cada agente Actor.
pub enum ActorMessage {
    /// Asigna la ejecución de una tarea con instrucciones y contexto previo.
    ExecuteTask {
        task_id: String,
        instructions: String,
        context_data: Option<String>,
        reply_to: oneshot::Sender<Result<String, String>>,
    },
    /// Solicita el cierre ordenado del actor.
    Shutdown,
}

/// Identificador y manipulador de comunicación concurrente hacia el Actor.
#[derive(Clone)]
pub struct AgentActorHandle {
    pub role: AgentRole,
    sender: mpsc::Sender<ActorMessage>,
    state: Arc<RwLock<AgentState>>,
}

impl AgentActorHandle {
    /// Envía una tarea para ser resuelta por el actor y espera asíncronamente el resultado.
    pub async fn execute_task(
        &self,
        task_id: impl Into<String>,
        instructions: impl Into<String>,
        context_data: Option<String>,
    ) -> Result<String, String> {
        let (reply_to, rx) = oneshot::channel();
        let msg = ActorMessage::ExecuteTask {
            task_id: task_id.into(),
            instructions: instructions.into(),
            context_data,
            reply_to,
        };

        self.sender
            .send(msg)
            .await
            .map_err(|e| format!("Failed to send task to actor {}: {e}", self.role))?;

        rx.await
            .map_err(|e| format!("Actor {} dropped reply channel: {e}", self.role))?
    }

    /// Consulta el estado operativo del actor en tiempo real sin bloquear el buzón de mensajes.
    pub fn get_status(&self) -> AgentState {
        self.state.read().unwrap().clone()
    }

    /// Cierra el actor de forma controlada.
    pub async fn shutdown(&self) {
        let _ = self.sender.send(ActorMessage::Shutdown).await;
    }
}

/// Estructura interna del Actor que corre en su propia tarea asíncrona dedicada.
pub struct AgentActor {
    role: AgentRole,
    engine: ReActEngine,
    receiver: mpsc::Receiver<ActorMessage>,
    state: Arc<RwLock<AgentState>>,
    event_bus: Option<EventBus>,
    system_prompt: String,
}

impl AgentActor {
    /// Spawnea un nuevo agente como Actor independiente en el runtime de Tokio.
    pub fn spawn(
        role: AgentRole,
        engine: ReActEngine,
        system_prompt: impl Into<String>,
        event_bus: Option<EventBus>,
        mailbox_capacity: usize,
    ) -> AgentActorHandle {
        let (sender, receiver) = mpsc::channel(mailbox_capacity);
        let state = Arc::new(RwLock::new(AgentState::Idle));

        let actor = Self {
            role,
            engine,
            receiver,
            state: Arc::clone(&state),
            event_bus,
            system_prompt: system_prompt.into(),
        };

        // El actor vive en su propio hilo asíncrono
        tokio::spawn(async move {
            actor.run().await;
        });

        AgentActorHandle {
            role,
            sender,
            state,
        }
    }

    async fn run(mut self) {
        while let Some(msg) = self.receiver.recv().await {
            match msg {
                ActorMessage::ExecuteTask {
                    task_id,
                    instructions,
                    context_data,
                    reply_to,
                } => {
                    *self.state.write().unwrap() = AgentState::Thinking;

                    if let Some(bus) = &self.event_bus {
                        bus.publish(AgentEvent::TaskStatusChanged {
                            task_id: task_id.clone(),
                            role: self.role.to_string(),
                            status: "InProgress".into(),
                        });
                    }

                    // Construir contexto para esta tarea
                    let mut context = AgentContext::new(self.role, &self.system_prompt);
                    if let Some(ctx_text) = context_data {
                        context.add_message(crate::agent::AgentMessage::user(format!(
                            "Contexto previo de otras tareas:\n{ctx_text}"
                        )));
                    }
                    context.add_message(crate::agent::AgentMessage::user(instructions));

                    // Ejecutar el bucle ReAct del agente
                    let result = self.engine.run(&mut context).await;

                    let response = match result {
                        Ok(res) => {
                            *self.state.write().unwrap() = AgentState::Completed;
                            if let Some(bus) = &self.event_bus {
                                bus.publish(AgentEvent::TaskStatusChanged {
                                    task_id,
                                    role: self.role.to_string(),
                                    status: "Completed".into(),
                                });
                            }
                            Ok(res.final_answer)
                        }
                        Err(err) => {
                            *self.state.write().unwrap() = AgentState::Failed {
                                reason: err.clone(),
                            };
                            if let Some(bus) = &self.event_bus {
                                bus.publish(AgentEvent::TaskStatusChanged {
                                    task_id,
                                    role: self.role.to_string(),
                                    status: format!("Failed: {err}"),
                                });
                            }
                            Err(err)
                        }
                    };

                    let _ = reply_to.send(response);
                }

                ActorMessage::Shutdown => {
                    break;
                }
            }
        }
    }
}
