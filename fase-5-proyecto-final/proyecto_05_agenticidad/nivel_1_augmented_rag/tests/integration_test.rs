use std::path::Path;
use nivel_1_augmented_rag::retriever::CodebaseRetriever;
use nivel_1_augmented_rag::AugmentedAgent;

#[tokio::test]
async fn test_retriever_builds_augmented_prompt_and_preserves_broken_sandbox() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let target = manifest_dir.parent().unwrap().join("sandbox/broken_service");

    // 1. Probar que el retriever lee el código real
    let code = CodebaseRetriever::retrieve_code_context(&target).unwrap();
    assert!(code.contains("fn calculate_tax"));
    assert!(code.contains("amount * 0.18"));

    // 2. Probar que construye el prompt aumentado
    let prompt = CodebaseRetriever::build_augmented_prompt(&target, "Corrige el error de tipos").unwrap();
    assert!(prompt.contains("CONTEXTO DEL CÓDIGO FUENTE"));
    assert!(prompt.contains("E0308"));
    assert!(prompt.contains("format!"));

    // 3. Probar que el agente responde con Rust
    let agent = AugmentedAgent::new(None);
    let (_, response) = agent.query_with_rag(&target, "Corrige el cálculo").await.unwrap();
    assert!(response.contains("fn calculate_tax"));
    assert!(response.contains("format!"));

    // 4. Probar que el sandbox físico sigue roto (cero modificaciones)
    let (ok, err) = AugmentedAgent::verify_sandbox(&target);
    assert!(!ok, "El sandbox debe seguir roto en Nivel 1");
    assert!(err.contains("error[E0308]"));
}
