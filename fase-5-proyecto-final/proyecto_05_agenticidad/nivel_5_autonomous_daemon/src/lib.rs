//! Nivel 5 de Agenticidad: Demonio Autónomo 24/7 y Auto-Sanación Proactiva (Always-On Daemon).
//! El pináculo de la autonomía: opera en segundo plano y se auto-despierta ante anomalías
//! sin requerir intervención humana (Zero-Human-in-the-Loop).

pub mod daemon;
pub mod healer;
pub mod watcher;

pub use daemon::{AutonomousDaemon, DaemonConfig};
pub use healer::{AutoHealer, IncidentReport};
pub use watcher::{HealthStatus, ServiceWatcher};
