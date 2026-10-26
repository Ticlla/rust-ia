//! ============================================================================
//! 🛠️ FASE 2 - CLASE 06: Laboratorio de Errores Comunes con Asincronía
//! ============================================================================
//!
//! Este laboratorio explora los 3 errores conceptuales y de código más comunes
//! que cometen los programadores al llegar de Python/JS a Rust asíncrono.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c06_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Asincronía Básica en Rust");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Olvidar hacer .await (El código fantasma)
    // =========================================================================
    println!("--- Error 1: Olvidar el .await (En Rust ¡NADA se ejecuta!) ---");
    println!("Si en Python o JS escribes:");
    println!("    hacer_peticion(); // Corre en el fondo o lanza una promesa");
    println!("\nPero si en Rust escribes:");
    println!("    guardar_checkpoint_en_s3(); // ❌ NO HACE NADA DE NADA!");
    println!("\n¿Qué dice el compilador de Rust?");
    println!("    warning: unused implementor of `Future` that must be used");
    println!("    note: futures do nothing unless you `.await` or poll them");
    println!("\n👉 EXPLICACIÓN: Una función async solo construye una receta (struct).");
    println!("   Hasta que no le pones '.await', la receta se destruye sin cocinarse.\n");

    // =========================================================================
    // ERROR COMÚN 2: Bloquear el Hilo con thread::sleep dentro de async
    // =========================================================================
    println!("--- Error 2: Usar thread::sleep() en código asíncrono ---");
    println!("Si dentro de una función 'async' escribes:");
    println!("    std::thread::sleep(Duration::from_secs(5)); // 💀 ANTIPATRÓN GRAVE!");
    println!("\n¿Por qué es un desastre en un servidor de IA?");
    println!("Porque un solo hilo del runtime asíncrono puede estar atendiendo");
    println!("a 10,000 usuarios a la vez en cooperativo.");
    println!("Si congelas el hilo del OS, ¡congelas a los 10,000 usuarios a la vez!");
    println!("👉 REGLA: En código async, usa timers asíncronos (como 'tokio::time::sleep').\n");

    // =========================================================================
    // ERROR COMÚN 3: Confundir Concurrencia con Paralelismo
    // =========================================================================
    println!("--- Error 3: Confundir Async con Multihilo (Rayon vs Tokio) ---");
    println!("* ¿Tienes 5 millones de números flotantes para calcular GELU? -> Usa RAYON (CPU-Bound).");
    println!("* ¿Tienes 5,000 usuarios esperando respuestas de red o base de datos? -> Usa ASYNC (I/O-Bound).");
    println!("👉 REGLA: Si la CPU suda haciendo matemáticas, usa Rayon.");
    println!("👉        Si la CPU duerme esperando bytes de la red, usa Async.\n");

    println!("✅ Todos los casos de error y conceptos asíncronos comprendidos.");
}
