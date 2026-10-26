//! Modelos de datos y DTOs para la API REST y streaming de la orquestación.

use crate::orchestrator::WorkflowReport;
use serde::{Deserialize, Serialize};

/// Solicitud para iniciar un flujo de trabajo orquestado.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowRequest {
    pub goal: String,
    pub workflow_id: Option<String>,
}

/// Respuesta tras la conclusión o aceptación del flujo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowResponse {
    pub workflow_id: String,
    pub status: String,
    pub report: WorkflowReport,
}

/// Estado de salud y métricas de la plataforma multi-agente.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub active_actors: usize,
    pub version: String,
}
