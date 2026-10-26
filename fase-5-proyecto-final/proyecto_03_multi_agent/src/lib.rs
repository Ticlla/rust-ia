//! # Autonomous Agent & Multi-Agent Orchestration Framework
//!
//! Framework de orquestación de agentes autónomos y sistemas multi-agente en Rust puro
//! basado en el **Modelo de Actores**, el patrón **ReAct** (Reasoning + Acting) y grafos de tareas (DAG).

pub mod agent;
pub mod api;
pub mod events;
pub mod orchestrator;
pub mod tools;

use agent::{AgentRole, DeterministicLlmBackend, ReActEngine};
use events::EventBus;
use orchestrator::{AgentActor, SupervisorOrchestrator};
use std::sync::Arc;
use tools::{CalculatorTool, RagSearchTool, RustCodeValidatorTool, ToolRegistry};

/// Inicializa el sistema completo de agentes con herramientas y bus de eventos usando el backend determinista.
pub fn create_default_system() -> (Arc<SupervisorOrchestrator>, EventBus) {
    create_system_with_backend(Arc::new(DeterministicLlmBackend))
}

/// Inicializa el sistema multi-agente con un backend de lenguaje personalizado (GatewayLlmBackend o DeterministicLlmBackend).
pub fn create_system_with_backend(
    llm_backend: Arc<dyn agent::LlmBackend>,
) -> (Arc<SupervisorOrchestrator>, EventBus) {
    let event_bus = EventBus::new(1024);

    // Registro de herramientas compartidas
    let mut tools = ToolRegistry::new();
    tools.register(Arc::new(CalculatorTool));
    tools.register(Arc::new(RagSearchTool));
    tools.register(Arc::new(RustCodeValidatorTool));

    // Spawneo de actores especializados (Modelo de Actores en Tokio)
    let researcher_engine = ReActEngine::new(llm_backend.clone(), tools.clone())
        .with_event_bus(event_bus.clone());
    let researcher = AgentActor::spawn(
        AgentRole::Researcher,
        researcher_engine,
        "Eres un Agente Investigador Técnico especializado en arquitecturas de IA y bajo nivel.",
        Some(event_bus.clone()),
        32,
    );

    let coder_engine = ReActEngine::new(llm_backend.clone(), tools.clone())
        .with_event_bus(event_bus.clone());
    let coder = AgentActor::spawn(
        AgentRole::Coder,
        coder_engine,
        "Eres un Agente Desarrollador especializado en escribir código Rust seguro, concurrente y modular.",
        Some(event_bus.clone()),
        32,
    );

    let reviewer_engine = ReActEngine::new(llm_backend.clone(), tools.clone())
        .with_event_bus(event_bus.clone());
    let reviewer = AgentActor::spawn(
        AgentRole::Reviewer,
        reviewer_engine,
        "Eres un Agente Auditor y Revisor de Código enfocado en memoria segura, robustez y calidad de tipos.",
        Some(event_bus.clone()),
        32,
    );

    let supervisor = Arc::new(SupervisorOrchestrator::new(
        researcher,
        coder,
        reviewer,
        event_bus.clone(),
    ));

    (supervisor, event_bus)
}
