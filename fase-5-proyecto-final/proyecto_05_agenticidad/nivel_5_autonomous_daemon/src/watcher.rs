//! Monitor de salud y detector proactivo de anomalías para el Demonio Autónomo.
//! Inspecciona de forma continua el estado de compilación del microservicio.

use std::path::Path;
use std::process::Command;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded(String),
}

pub struct ServiceWatcher {
    target_dir: std::path::PathBuf,
}

impl ServiceWatcher {
    pub fn new(target_dir: &Path) -> Self {
        Self {
            target_dir: target_dir.to_path_buf(),
        }
    }

    /// Comprueba la salud del microservicio ejecutando 'cargo check' en el sandbox
    pub fn check_health(&self) -> HealthStatus {
        let output = Command::new("cargo")
            .arg("check")
            .current_dir(&self.target_dir)
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    HealthStatus::Healthy
                } else {
                    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    HealthStatus::Degraded(stderr)
                }
            }
            Err(e) => HealthStatus::Degraded(format!("Error invocando el compilador: {e}")),
        }
    }

    /// Comprueba si las pruebas unitarias pasan en disco
    pub fn verify_tests(&self) -> bool {
        Command::new("cargo")
            .arg("test")
            .current_dir(&self.target_dir)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}
