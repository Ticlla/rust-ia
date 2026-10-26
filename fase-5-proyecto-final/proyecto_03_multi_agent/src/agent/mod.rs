//! Módulo de agentes individuales y bucle ReAct.

pub mod llm_client;
pub mod react;
pub mod state;

pub use llm_client::{DeterministicLlmBackend, LlmBackend, LlmDecision};
pub use react::{ReActEngine, ReActResult};
pub use state::{AgentContext, AgentMessage, AgentRole, AgentState, ReActStep};
