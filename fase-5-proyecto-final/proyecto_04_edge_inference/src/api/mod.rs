//! Módulo de interfaz API HTTP (Axum) para el microservicio de inferencia edge.

pub mod handlers;
pub mod models;
pub mod routes;

pub use handlers::AppState;
pub use routes::create_router;
