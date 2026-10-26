//! Módulo API REST y Server-Sent Events.

pub mod handlers;
pub mod models;
pub mod routes;

pub use handlers::AppState;
pub use models::{HealthResponse, WorkflowRequest, WorkflowResponse};
pub use routes::build_router;
