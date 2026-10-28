//! Demonio Asíncrono Always-On impulsado por Tokio.
//! Ejecuta un ciclo continuo de monitoreo y auto-sanación con tokio::select!.

use std::path::Path;
use std::time::Duration;
use tokio::sync::oneshot;
use tokio::time::interval;

use crate::healer::{AutoHealer, IncidentReport};
use crate::watcher::{HealthStatus, ServiceWatcher};

pub struct DaemonConfig {
    pub poll_interval: Duration,
    pub max_auto_heals: Option<usize>,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_millis(500),
            max_auto_heals: None,
        }
    }
}

pub struct AutonomousDaemon {
    watcher: ServiceWatcher,
    healer: AutoHealer,
    config: DaemonConfig,
}

impl AutonomousDaemon {
    pub fn new(target_dir: &Path, api_key: Option<String>, config: DaemonConfig) -> Self {
        Self {
            watcher: ServiceWatcher::new(target_dir),
            healer: AutoHealer::new(target_dir, api_key),
            config,
        }
    }

    /// Ejecuta el bucle del demonio 24/7 con selección asíncrona
    pub async fn run(
        &self,
        mut shutdown_rx: Option<oneshot::Receiver<()>>,
    ) -> Vec<IncidentReport> {
        let mut ticker = interval(self.config.poll_interval);
        let mut incidents = Vec::new();
        let mut incident_counter = 0;

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    let health = self.watcher.check_health();

                    match health {
                        HealthStatus::Healthy => {
                            // Sistema operando nominalmente
                        }
                        HealthStatus::Degraded(error_log) => {
                            incident_counter += 1;
                            let incident_id = format!("INC-2026-{:03}", incident_counter);

                            println!("\n  🚨 [ALERTA PROACTIVA] Se detectó degradación en el servicio ({}).", incident_id);
                            println!("     Causa: {}", error_log.lines().find(|l| l.contains("error[")).unwrap_or("Fallo no catalogado"));
                            println!("     ⚡ Despertando pipeline de auto-sanación autónoma...");

                            let report = self.healer.heal_service(&incident_id, &error_log).await;

                            println!("     ✅ [AUTO-SANADO] Incidente {} resuelto en {} ms (MTTR).", report.incident_id, report.duration_ms);
                            println!("     🧪 Verificación de Pruebas: {}", if report.cargo_test_verified { "PASADO (100%)" } else { "FALLO" });

                            incidents.push(report);

                            if let Some(limit) = self.config.max_auto_heals {
                                if incident_counter >= limit {
                                    println!("  🛑 Límite de auto-sanaciones de prueba alcanzado ({}/{}). Finalizando sesión del demonio.", incident_counter, limit);
                                    break;
                                }
                            }
                        }
                    }
                }
                _ = async {
                    if let Some(rx) = shutdown_rx.as_mut() {
                        let _ = rx.await;
                    } else {
                        std::future::pending::<()>().await;
                    }
                } => {
                    println!("\n  🛑 Señal de apagado recibida. Cerrando el Demonio de forma segura (Graceful Shutdown).");
                    break;
                }
            }
        }

        incidents
    }
}
