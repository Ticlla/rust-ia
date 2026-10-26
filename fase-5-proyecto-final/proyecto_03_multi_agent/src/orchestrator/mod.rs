//! Módulo de orquestación multi-agente, modelo de actores y grafos de tareas (DAG).

pub mod actor;
pub mod dag;
pub mod supervisor;

pub use actor::{ActorMessage, AgentActor, AgentActorHandle};
pub use dag::{TaskGraph, TaskNode, TaskStatus};
pub use supervisor::{SupervisorOrchestrator, WorkflowReport};
