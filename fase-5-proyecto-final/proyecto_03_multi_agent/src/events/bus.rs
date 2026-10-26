//! Bus de eventos reactivo basado en canales `tokio::sync::broadcast` para observabilidad en tiempo real.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::broadcast;

/// Evento granular emitido durante la ejecución de los agentes y flujos de trabajo.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event_type", content = "payload")]
pub enum AgentEvent {
    /// Inicio de un flujo de trabajo orquestado.
    WorkflowStarted {
        workflow_id: String,
        goal: String,
        timestamp_ms: u64,
    },
    /// Paso de razonamiento interno de un agente (ReAct Thought).
    AgentThinking {
        agent_role: String,
        thought: String,
    },
    /// Disparo de invocación de una herramienta con sus parámetros.
    ToolCallStarted {
        agent_role: String,
        tool_name: String,
        arguments: serde_json::Value,
    },
    /// Finalización de la ejecución de una herramienta con resultados o error.
    ToolCallCompleted {
        agent_role: String,
        tool_name: String,
        success: bool,
        output: serde_json::Value,
        duration_ms: u64,
    },
    /// Paso de mensajes o contexto entre agentes actores.
    AgentMessageSent {
        from_agent: String,
        to_agent: String,
        content: String,
    },
    /// Dictamen de revisión y auditoría de código o investigación.
    ReviewOutcome {
        approved: bool,
        score: f32,
        feedback: String,
    },
    /// Cambio de estado en una subtarea del grafo de tareas (DAG).
    TaskStatusChanged {
        task_id: String,
        role: String,
        status: String,
    },
    /// Finalización exitosa del flujo completo de trabajo.
    WorkflowCompleted {
        workflow_id: String,
        summary: String,
        total_duration_ms: u64,
    },
    /// Fracaso o interrupción del flujo de trabajo.
    WorkflowFailed {
        workflow_id: String,
        error: String,
    },
}

/// Bus de eventos de alta concurrencia que permite a múltiples suscriptores (SSE, logs, métricas)
/// recibir eventos en tiempo real con cero contención de bloqueo.
#[derive(Clone)]
pub struct EventBus {
    sender: Arc<broadcast::Sender<AgentEvent>>,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(1024)
    }
}

impl EventBus {
    /// Inicializa un nuevo bus de eventos con una capacidad fija de búfer circular.
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender: Arc::new(sender),
        }
    }

    /// Publica un evento en el bus para todos los suscriptores activos.
    pub fn publish(&self, event: AgentEvent) {
        // Ignora el error si no hay receptores activos en este instante
        let _ = self.sender.send(event);
    }

    /// Crea un nuevo receptor (suscriptor) para consumir eventos en tiempo real.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.sender.subscribe()
    }
}
