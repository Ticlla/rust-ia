//! Definición de roles especializados para la arquitectura Multi-Agente (Nivel 4).
//! Aplica el principio de Mínimo Privilegio (Principle of Least Privilege).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentRole {
    /// Supervisor: Genera el plan DAG, asigna tareas y consolida el veredicto final.
    Supervisor,
    /// Analista: Inspecciona el código fuente y el compilador. Acceso de solo lectura.
    Analyzer,
    /// Desarrollador: Escribe parches de código basado en el diagnóstico.
    Coder,
    /// Auditor / QA: Ejecuta pruebas y valida que la solución no vulnere o altere el test suite.
    Auditor,
}

impl AgentRole {
    pub fn name(&self) -> &'static str {
        match self {
            AgentRole::Supervisor => "SupervisorOrchestrator",
            AgentRole::Analyzer => "AnalyzerAgent",
            AgentRole::Coder => "CoderAgent",
            AgentRole::Auditor => "AuditorQAAgent",
        }
    }

    pub fn emoji(&self) -> &'static str {
        match self {
            AgentRole::Supervisor => "👔",
            AgentRole::Analyzer => "🔍",
            AgentRole::Coder => "💻",
            AgentRole::Auditor => "🛡️",
        }
    }

    /// System Prompt ultra-especializado por rol para evitar alucinaciones y dispersión
    pub fn system_prompt(&self) -> &'static str {
        match self {
            AgentRole::Supervisor => {
                "Eres el SupervisorOrchestrator del Nivel 4. Tu rol es desglosar la meta en un DAG \
                de tareas especializadas, delegar a los agentes correspondientes y verificar que la misión \
                se cumpla bajo estándares de calidad estricta."
            }
            AgentRole::Analyzer => {
                "Eres el AnalyzerAgent (Experto en Diagnóstico de Rust). \
                Tu rol es EXCLUSIVAMENTE de LECTURA y DIAGNÓSTICO. \
                Tienes acceso a las herramientas 'read_file' y 'run_cargo_check'. \
                NO tienes permitido escribir ni modificar archivos. \
                Examina el código y el error de compilación rustc, identifica el error exacto (ej: E0308) \
                y emite un informe estructurado con la causa raíz y las instrucciones de reparación."
            }
            AgentRole::Coder => {
                "Eres el CoderAgent (Desarrollador Rust Especializado). \
                Tu rol es implementar la solución recomendada por el AnalyzerAgent escribiendo el archivo con 'write_file'. \
                Debes escribir en 'src/main.rs' el código Rust completo y válido donde 'calculate_tax' retorne format!(\"{:.2}\", amount * 0.18). \
                NO modifiques las aserciones de prueba en el módulo tests. \
                Llama a 'write_file' con relative_path: 'src/main.rs' y el contenido Rust limpio."
            }
            AgentRole::Auditor => {
                "Eres el AuditorQAAgent (Auditor Independiente de Calidad y Seguridad). \
                Tu rol es verificar con 'run_cargo_test' y 'read_file' que el código corregido \
                compile y que todas las pruebas unitarias pasen al 100%. \
                Inspecciona que el CoderAgent no haya alterado las pruebas para 'hacer trampa'. \
                NO tienes acceso a 'write_file'. Emite un veredicto final: APROBADO o RECHAZADO."
            }
        }
    }

    /// Lista de herramientas autorizadas para este rol (Control de Acceso RBAC)
    pub fn permitted_tools(&self) -> Vec<&'static str> {
        match self {
            AgentRole::Supervisor => vec![],
            AgentRole::Analyzer => vec!["read_file", "run_cargo_check"],
            AgentRole::Coder => vec!["read_file", "write_file"],
            AgentRole::Auditor => vec!["read_file", "run_cargo_test"],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: AgentRole,
    pub to: AgentRole,
    pub task_id: String,
    pub payload: String,
}
