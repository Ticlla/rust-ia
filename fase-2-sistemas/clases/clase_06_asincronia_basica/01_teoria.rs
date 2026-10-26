//! ============================================================================
//! 🦀 FASE 2 - CLASE 06: Asincronía Básica (`async` / `await` y `Future`) en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. I/O-Bound vs CPU-Bound: ¿Por qué necesitamos asincronía si ya tenemos hilos?
//! 2. Qué es una `Future` en Rust: Una máquina de estados perezosa (Lazy).
//! 3. La regla de oro: En Rust un `Future` no hace NADA hasta que le haces `.await`.
//! 4. Concurrencia cooperativa con `futures::join!`: Descargar 3 datasets a la vez.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c06_teoria

use futures::executor::block_on;
use futures::join;
use std::time::Duration;

/// Simula una llamada de red asíncrona a un proveedor de embeddings (API externa).
/// Nota: 'async fn' transforma esta función en una máquina de estados que implementa 'Future'.
async fn consultar_api_embeddings(modelo: &str, texto: &str) -> Vec<f32> {
    println!("  [API {modelo}] 📡 Iniciando petición para: \"{texto}\"...");

    // Simula latencia de red (en sistemas reales se usa un timer asíncrono):
    std::thread::sleep(Duration::from_millis(50));

    println!("  [API {modelo}] ✅ Respuesta recibida con éxito.");
    vec![0.15, 0.42, 0.88]
}

/// Simula la descarga asíncrona de un lote de entrenamiento desde un bucket S3.
async fn descargar_lote_s3(lote_id: usize) -> String {
    println!("  [S3 Storage] 📥 Descargando lote #{lote_id}...");
    std::thread::sleep(Duration::from_millis(70));
    println!("  [S3 Storage] 📦 Lote #{lote_id} listo en búfer.");
    format!("datos_lote_{lote_id}")
}

fn main() {
    println!("============================================================");
    println!("⏳ FASE 2 - CLASE 06: Asincronía Básica (async/await)");
    println!("============================================================\n");

    // =========================================================================
    // 1. LA NATURALEZA PEREZOSA (LAZY) DE LAS FUTURES EN RUST
    // =========================================================================
    println!("--- 1. Demostración: Los Futures son 100% Perezosos (Lazy) ---");

    // 💡 ATENCIÓN: Al llamar a esta función 'async', ¡NO SE EJECUTA NADA!
    // Solo crea la estructura de la máquina de estados en la memoria:
    let futuro_perezoso = consultar_api_embeddings("text-embedding-3", "Hola Mundo");

    println!("👉 Acabamos de invocar 'consultar_api_embeddings()'.");
    println!("👉 Fíjate en la consola: ¡NO hay ningún mensaje de red todavía!");
    println!("👉 En Rust, los Futures NO hacen nada hasta que los empujas a correr.\n");

    // Ahora sí, ejecutamos el futuro usando el ejecutor 'block_on':
    println!("--- Empujando al Future a ejecutarse cargo run --bin c06_teoriacon block_on() ---");
    let vector = block_on(futuro_perezoso);
    println!("Resultado del embedding: {:?}\n", vector);

    // =========================================================================
    // 2. CONCURRENCIA ASÍNCRONA COOPERATIVA CON join!
    // =========================================================================
    println!("--- 2. Concurrencia Asíncrona con futures::join! ---");
    println!("Consultando 2 APIs externas y descargando 1 lote S3 concurrentemente:");

    let pipeline_asincrono = async {
        // 'join!' corre múltiples futures de forma concurrente cooperativa:
        let (emb_1, emb_2, lote) = join!(
            consultar_api_embeddings("Titan-Embeddings", "Rust en IA"),
            consultar_api_embeddings("Cohere-v3", "Transformers y Rayon"),
            descargar_lote_s3(42)
        );

        println!("\n--- 3. Consolidación de Resultados en el Pipeline ---");
        println!("Emb Titan   : {:?}", emb_1);
        println!("Emb Cohere  : {:?}", emb_2);
        println!("Lote S3     : {}", lote);
    };

    // Corremos el pipeline asíncrono completo:
    block_on(pipeline_asincrono);

    println!("\n✅ Fin de la demostración de Asincronía Básica.");
}
