use std::path::Path;
use nivel_0_zero_shot::ZeroShotAgent;

#[tokio::test]
async fn test_zero_shot_generates_text_without_altering_sandbox() {
    let agent = ZeroShotAgent::new(None);
    let prompt = "¿Cómo soluciono un error de tipos en Rust?";
    let response = agent.query(prompt).await.unwrap();

    // El modelo responde texto
    assert!(!response.is_empty());
    assert!(response.contains("Rust") || response.contains("String") || response.contains("format!"));

    // Pero no altera el sandbox
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir.parent().unwrap().join("sandbox/broken_service");

    let (ok, err) = ZeroShotAgent::verify_sandbox(&target);
    assert!(!ok, "El sandbox debe seguir roto en Nivel 0");
    assert!(err.contains("error[E0308]") || err.contains("mismatched types"));
}
