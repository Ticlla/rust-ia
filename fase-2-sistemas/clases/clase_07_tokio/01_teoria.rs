//! ============================================================================
//! 🦀 FASE 2 - CLASE 07: Runtime Asíncrono con `Tokio` en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Qué es el Runtime de `Tokio` y la macro `#[tokio::main]`.
//! 2. Tareas Verdes (*Green Tasks*) con `tokio::spawn`: miles de tareas en RAM mínima.
//! 3. El secreto de la IA en Rust: `tokio::task::spawn_blocking` para separar I/O de Tensores.
//! 4. Canales asíncronos con `tokio::sync::mpsc`.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c07_teoria

use std::time::Duration;
use tokio::time::sleep;

/// Simula una llamada I/O de red (espera pasiva sin usar CPU).
async fn recibir_prompt_usuario(id: usize) -> String {
    println!("  [Red] 🌐 Usuario #{id} conectado. Esperando prompt por socket...");
    // 💡 'sleep' asíncrono: cede el turno cooperativamente, NO congela el hilo:
    sleep(Duration::from_millis(50)).await;
    format!("Prompt del usuario #{id}: ¿Qué es Tokio?")
}

/// Simula un cómputo pesado de IA (CPU-Bound: multiplicación matricial).
/// Esta función es sincrónica normal porque hace sudar al CPU.
fn computar_inferencia_pesada(prompt: String) -> String {
    // Simula cálculo intensivo de CPU:
    let mut hash: u64 = 0;
    for b in prompt.bytes() {
        hash = hash.wrapping_mul(31).wrapping_add(b as u64);
    }
    // Simula retardo de cálculo matricial:
    std::thread::sleep(Duration::from_millis(30));
    format!("Inferencia lista [Hash: {:x}]", hash)
}

#[tokio::main]
async fn main() {
    println!("============================================================");
    println!("⚡ FASE 2 - CLASE 07: El Runtime Asíncrono de Tokio");
    println!("============================================================\n");

    println!("--- 1. Lanzando Tareas Asíncronas Ligeras (tokio::spawn) ---");

    let mut handles = Vec::new();

    for id in 1..=4 {
        // 💡 'tokio::spawn' lanza una tarea asíncrona independiente en el Thread Pool de Tokio:
        let h = tokio::spawn(async move {
            // FASE 1: I/O Asíncrono (Red)
            let prompt = recibir_prompt_usuario(id).await;

            // FASE 2: Cómputo CPU-Bound (IA / Tensores)
            // 💡 REGLA DE ORO EN IA:
            // Usamos 'spawn_blocking' para no congelar las tareas de red de los otros usuarios!
            let resultado = tokio::task::spawn_blocking(move || {
                computar_inferencia_pesada(prompt)
            })
            .await
            .unwrap();

            println!("  [Servicio] 🎯 Usuario #{id} procesado: {resultado}");
        });

        handles.push(h);
    }

    // Esperamos a que todas las tareas concurrentes terminen:
    for h in handles {
        h.await.unwrap();
    }

    // =========================================================================
    // 2. CANALES ASÍNCRONOS (tokio::sync::mpsc)
    // =========================================================================
    println!("\n--- 2. Streaming de Tokens con Canales Asíncronos (mpsc) ---");

    // Canal asíncrono con buffer de 10 mensajes:
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(10);

    // Tarea productora (Simula un LLM generando tokens palabra por palabra):
    tokio::spawn(async move {
        let tokens = ["Rust", "es", "el", "futuro", "de", "la", "IA."];
        for token in tokens {
            sleep(Duration::from_millis(20)).await;
            tx.send(token.to_string()).await.unwrap();
        }
    });

    // Tarea consumidora (Simula enviar tokens por Server-Sent Events / SSE al cliente):
    print!("  [Streaming al cliente]: ");
    while let Some(token) = rx.recv().await {
        print!("{token} ");
    }
    println!();

    println!("\n✅ Fin de la Clase 07 de Tokio.");
}
