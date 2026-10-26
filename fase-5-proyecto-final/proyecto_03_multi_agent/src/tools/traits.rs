//! Definición del trait `Tool` y tipos de retorno para herramientas tipadas.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Error resultante de la ejecución o validación de una herramienta.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ToolError {
    /// Parámetros inválidos según el esquema JSON.
    InvalidParameters(String),
    /// Tiempo límite de ejecución excedido (Sandbox timeout).
    Timeout(String),
    /// Fallo en la lógica interna de ejecución de la herramienta.
    ExecutionFailed(String),
    /// La herramienta solicitada no existe en el registro.
    NotFound(String),
}

impl fmt::Display for ToolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters(msg) => write!(f, "Invalid parameters: {msg}"),
            Self::Timeout(msg) => write!(f, "Execution timeout: {msg}"),
            Self::ExecutionFailed(msg) => write!(f, "Execution failed: {msg}"),
            Self::NotFound(msg) => write!(f, "Tool not found: {msg}"),
        }
    }
}

impl std::error::Error for ToolError {}

/// Salida tipada producida por la ejecución de una herramienta.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolOutput {
    /// Nombre de la herramienta que produjo la salida.
    pub tool_name: String,
    /// Éxito o fracaso funcional de la operación.
    pub success: bool,
    /// Contenido textual o estructurado del resultado.
    pub result: serde_json::Value,
    /// Metadatos adicionales (tiempo de cómputo, citas, telemetría).
    pub metadata: Option<serde_json::Value>,
}

impl ToolOutput {
    /// Construye una salida exitosa con un valor JSON.
    pub fn success(tool_name: impl Into<String>, result: serde_json::Value) -> Self {
        Self {
            tool_name: tool_name.into(),
            success: true,
            result,
            metadata: None,
        }
    }

    /// Construye una salida exitosa con metadatos.
    pub fn success_with_meta(
        tool_name: impl Into<String>,
        result: serde_json::Value,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            success: true,
            result,
            metadata: Some(metadata),
        }
    }

    /// Construye una salida fallida con un mensaje explicativo.
    pub fn failure(tool_name: impl Into<String>, error_message: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            success: false,
            result: serde_json::json!({ "error": error_message.into() }),
            metadata: None,
        }
    }
}

/// Contrato asíncrono y seguro para hilos (`Send + Sync`) para todas las herramientas del agente.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Nombre canónico e identificador único de la herramienta (ej. `"rag_search"`).
    fn name(&self) -> &str;

    /// Descripción funcional de lo que hace la herramienta y cuándo invocarla.
    fn description(&self) -> &str;

    /// Esquema JSON Schema para la validación estricta de parámetros de entrada.
    fn parameters_schema(&self) -> serde_json::Value;

    /// Ejecución asíncrona de la herramienta con argumentos JSON.
    async fn execute(&self, arguments: serde_json::Value) -> Result<ToolOutput, ToolError>;
}
