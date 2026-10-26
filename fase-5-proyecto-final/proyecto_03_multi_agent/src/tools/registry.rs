//! Registro dinámico y seguro de herramientas con control de timeout (Sandbox).

use super::traits::{Tool, ToolError, ToolOutput};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

/// Registro centralizado de herramientas disponibles para los agentes.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
    default_timeout: Duration,
}

impl ToolRegistry {
    /// Inicializa un registro vacío con un timeout por defecto (10 segundos).
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            default_timeout: Duration::from_secs(10),
        }
    }

    /// Configura el timeout de ejecución global para el registro.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Registra una nueva herramienta tipada en el inventario.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Obtiene una referencia compartida a una herramienta por su nombre.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Retorna la cantidad de herramientas registradas.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Verifica si el registro está vacío.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Retorna los nombres de todas las herramientas registradas.
    pub fn list_tool_names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Ejecuta una herramienta con aislamiento de timeout (Sandbox de tiempo).
    pub async fn execute(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ToolError::NotFound(format!("Tool '{name}' not found in registry")))?;

        match timeout(self.default_timeout, tool.execute(arguments)).await {
            Ok(result) => result,
            Err(_) => Err(ToolError::Timeout(format!(
                "Tool '{name}' exceeded maximum execution timeout of {:?}",
                self.default_timeout
            ))),
        }
    }

    /// Exporta las herramientas al formato estándar de OpenAI Functions / Tools (`type: "function"`).
    pub fn to_openai_tools_schema(&self) -> serde_json::Value {
        let tools_list: Vec<serde_json::Value> = self
            .tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name(),
                        "description": tool.description(),
                        "parameters": tool.parameters_schema(),
                    }
                })
            })
            .collect();

        serde_json::Value::Array(tools_list)
    }
}
