//! ============================================================================
//! 🛠️ FASE 2 - CLASE 10: Antipatrones y Errores de Arquitectura en Servidores de IA
//! ============================================================================
//!
//! Este laboratorio resume los 4 errores arquitectónicos más costosos que tumban
//! servidores de IA en producción y cómo nuestra arquitectura los previene.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c10_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ARQUITECTURA: Antipatrones en Servidores IA");
    println!("============================================================\n");

    // =========================================================================
    // ANTIPATRÓN 1: El Reactor Asfixiado (Starving the Tokio Reactor)
    // =========================================================================
    println!("--- 1. El Reactor Asfixiado (Cálculo CPU directo en Axum) ---");
    println!("El error: Ejecutar multiplicación de matrices de 500 ms directamente en el async fn.");
    println!("El síntoma: Las peticiones de salud (GET /health) de Kubernetes fallan por timeout.");
    println!("           Kubernetes cree que el contenedor murió y lo reinicia continuamente.");
    println!("👉 LA SOLUCIÓN DEL PROYECTO: Envolver todo cómputo numérico con:");
    println!("   tokio::task::spawn_blocking(move || {{ modelo.predecir_paralelo(...) }}).await\n");

    // =========================================================================
    // ANTIPATRÓN 2: Canales sin Contrapresión (Unbounded Channels)
    // =========================================================================
    println!("--- 2. Canales sin Límite (Fuga OOM por falta de Backpressure) ---");
    println!("El error: Usar 'mpsc::unbounded_channel()' para el DataLoader.");
    println!("El síntoma: El disco o la red descargan lotes más rápido de lo que el modelo infiere.");
    println!("           La cola acumula 100,000 lotes en RAM hasta que el servidor crashea por OOM.");
    println!("👉 LA SOLUCIÓN DEL PROYECTO: Usar siempre canales con buffer finito:");
    println!("   mpsc::channel(10) // Pausa al emisor con .send().await si la cola se llena.\n");

    // =========================================================================
    // ANTIPATRÓN 3: Deadlocks por Retener Guards a través de await
    // =========================================================================
    println!("--- 3. Retener Cerrojos a través de puntos .await ---");
    println!("El error: Adquirir un lock de escritura (RwLock) y hacer una llamada de red con .await.");
    println!("El síntoma: Todos los lectores se quedan congelados indefinidamente.");
    println!("👉 LA SOLUCIÓN DEL PROYECTO: Limitar el alcance {{ }} para soltar el cerrojo de inmediato:\n");
    println!("   let version = {{ let mut m = state.modelo.write().unwrap(); m.actualizar(); m.version }};");
    println!("   notificar_por_red(version).await; // 💡 El cerrojo ya fue liberado arriba!\n");

    // =========================================================================
    // ANTIPATRÓN 4: Fugas de Memoria en Tensores
    // =========================================================================
    println!("--- 4. Fugas de Memoria en Tensores Contiguos ---");
    println!("El error: Asignar búferes en heap o VRAM y no implementar RAII.");
    println!("👉 LA SOLUCIÓN DEL PROYECTO: El trait 'Drop' garantiza que 'dealloc' se ejecute");
    println!("   en el instante exacto en que el TensorBuffer sale de ámbito.\n");

    println!("✅ Todos los principios de arquitectura de sistemas de IA dominados.");
}
