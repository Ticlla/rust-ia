use nivel_5_autonomous_daemon::{
    AutoHealer, AutonomousDaemon, DaemonConfig, HealthStatus, ServiceWatcher,
};
use std::path::Path;
use std::time::Duration;
use tokio::sync::oneshot;

#[tokio::test]
async fn test_watcher_detects_healthy_and_degraded() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    let healer = AutoHealer::new(&target, None);
    let watcher = ServiceWatcher::new(&target);

    // 1. Inyectar fallo E0308
    healer.inject_corruption().unwrap();
    let status = watcher.check_health();
    assert!(
        matches!(status, HealthStatus::Degraded(_)),
        "El watcher debe detectar la degradación"
    );

    // 2. Auto-sanar
    let report = healer.heal_service("TEST-001", "error[E0308]").await;
    assert!(
        report.cargo_test_verified,
        "El healer debe corregir el código"
    );

    let status_healed = watcher.check_health();
    assert_eq!(
        status_healed,
        HealthStatus::Healthy,
        "El watcher debe confirmar el estado saludable"
    );

    // 3. Dejar el sandbox listo
    healer.inject_corruption().unwrap();
}

#[tokio::test]
async fn test_daemon_proactive_auto_healing_without_human() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir
        .parent()
        .unwrap()
        .join("sandbox/broken_service");

    let healer = AutoHealer::new(&target, None);
    healer.inject_corruption().unwrap();

    let config = DaemonConfig {
        poll_interval: Duration::from_millis(50),
        max_auto_heals: Some(1),
    };

    let daemon = AutonomousDaemon::new(&target, None, config);
    let (tx, rx) = oneshot::channel();

    // Ejecutar demonio en tarea de Tokio
    let daemon_handle = tokio::spawn(async move { daemon.run(Some(rx)).await });

    let incidents = daemon_handle.await.unwrap();
    let _ = tx; // Mantener vivo el canal

    assert_eq!(
        incidents.len(),
        1,
        "El demonio debe haber auto-sanado 1 incidente"
    );
    assert!(
        incidents[0].cargo_test_verified,
        "El incidente debe quedar verificado con cargo test"
    );
    assert_eq!(incidents[0].status, "RESOLVED");

    // Limpieza
    healer.inject_corruption().unwrap();
}
