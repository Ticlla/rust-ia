//! Módulo de herramientas tipadas (Tool Calling) para agentes autónomos.

pub mod builtin;
pub mod registry;
pub mod traits;

pub use builtin::{CalculatorTool, RagSearchTool, RustCodeValidatorTool};
pub use registry::ToolRegistry;
pub use traits::{Tool, ToolError, ToolOutput};
