//! Estados, roles y estructuras de memoria de contexto para agentes.

use serde::{Deserialize, Serialize};

/// Rol funcional asignado a un agente dentro de la orquestación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Descompone metas, asigna tareas y consolida el reporte final.
    Supervisor,
    /// Busca conocimiento en RAG, resume especificaciones y analiza requerimientos.
    Researcher,
    /// Escribe algoritmos, arquitecturas en Rust y estructuras de datos.
    Coder,
    /// Audita código, detecta vulnerabilidades y valida los criterios de aceptación.
    Reviewer,
}

impl AgentRole {
    /// Nombre en cadena legible del rol.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Supervisor => "Supervisor",
            Self::Researcher => "Researcher",
            Self::Coder => "Coder",
            Self::Reviewer => "Reviewer",
        }
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Estado del autómata finito (FSM) de un agente individual durante el ciclo ReAct.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    /// En reposo esperando instrucciones.
    Idle,
    /// Generando razonamiento interno (Thought).
    Thinking,
    /// Ejecutando una herramienta externa tipada (Action).
    CallingTool { tool_name: String },
    /// Procesando el resultado o error devuelto por la herramienta (Observation).
    Observing,
    /// Evaluando el avance hacia el objetivo y planificando el siguiente paso (Reflection).
    Reflecting,
    /// Meta completada exitosamente.
    Completed,
    /// Error irrecuperable o límite de iteraciones excedido.
    Failed { reason: String },
}

/// Mensaje en el historial conversacional y de razonamiento del agente.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentMessage {
    /// Rol emisor ("system", "user", "assistant", "tool").
    pub role: String,
    /// Contenido textual del mensaje.
    pub content: String,
    /// Nombre de la herramienta (si el rol es "tool").
    pub tool_name: Option<String>,
}

impl AgentMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: content.into(),
            tool_name: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: content.into(),
            tool_name: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            tool_name: None,
        }
    }

    pub fn tool(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_name: Some(tool_name.into()),
        }
    }
}

/// Paso unitario registrado dentro del bucle ReAct.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReActStep {
    pub iteration: usize,
    pub thought: String,
    pub action: Option<(String, serde_json::Value)>,
    pub observation: Option<serde_json::Value>,
    pub reflection: Option<String>,
}

/// Contexto y memoria a corto plazo del agente durante una tarea.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentContext {
    pub role: AgentRole,
    pub system_prompt: String,
    pub messages: Vec<AgentMessage>,
    pub steps: Vec<ReActStep>,
    pub max_iterations: usize,
}

impl AgentContext {
    pub fn new(role: AgentRole, system_prompt: impl Into<String>) -> Self {
        let prompt_str = system_prompt.into();
        Self {
            role,
            system_prompt: prompt_str.clone(),
            messages: vec![AgentMessage::system(prompt_str)],
            steps: Vec::new(),
            max_iterations: 8,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn add_message(&mut self, msg: AgentMessage) {
        self.messages.push(msg);
    }

    pub fn record_step(&mut self, step: ReActStep) {
        self.steps.push(step);
    }
}
