//! ============================================================================
//! 🛠️ FASE 2 - CLASE 01: Laboratorio de Errores Comunes con Hilos
//! ============================================================================
//!
//! Este laboratorio te enseña a interpretar los errores de concurrencia que
//! el compilador de Rust detecta en tiempo de compilación para salvarte de
//! corrupciones de memoria y caídas en producción.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c01_errores

fn main() {
    println!("========================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Hilos y Concurrencia en Rust");
    println!("========================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Olvidar la palabra clave 'move'
    // =========================================================================
    println!("--- Error 1: Olvidar 'move' al usar datos en un hilo ---");
    println!("Si intentas escribir:");
    println!("    let datos = vec![1.0, 2.0];");
    println!("    thread::spawn(|| println!(\"{{:?}}\", datos)); // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque el closure intenta PEDIR PRESTADO '&datos'. Si 'main' termina");
    println!("antes que el hilo, 'datos' se destruiría en RAM dejando al hilo apuntando a basura.");
    println!("👉 MENSAJE DE RUST: 'closure may outlive the current function, but it borrows data'");
    println!("👉 SOLUCIÓN: Usar 'thread::spawn(move || ...)' para transferir la propiedad.\n");

    // =========================================================================
    // ERROR COMÚN 2: Intentar usar una variable DESPUÉS de moverla al hilo
    // =========================================================================
    println!("--- Error 2: Usar una variable después de moverla al hilo ---");
    println!("Si intentas escribir:");
    println!("    let pesos = vec![0.5, 0.8];");
    println!("    thread::spawn(move || procesar(pesos));");
    println!("    println!(\"{{:?}}\", pesos); // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque la propiedad de 'pesos' ya viajó al otro hilo. En 'main' ya no existe.");
    println!("👉 MENSAJE DE RUST: 'borrow of moved value: pesos'");
    println!("👉 SOLUCIÓN: Si necesitas compartir datos entre varios hilos sin perderlos,");
    println!("             usaremos 'Arc<T>' (Atomic Reference Counting) en la Clase 03.\n");

    // =========================================================================
    // ERROR COMÚN 3: Olvidar hacer '.join()' (Hilos zombis / Hilos cortados)
    // =========================================================================
    println!("--- Error 3: No hacer .join() sobre el JoinHandle ---");
    println!("Si lanzas un hilo y 'main' llega a su final sin hacer '.join()',");
    println!("el Sistema Operativo MATA inmediatamente a todos los hilos secundarios,");
    println!("dejando el cómputo de tu modelo a la mitad sin que termine.");
    println!("👉 SOLUCIÓN: Guarda siempre los 'JoinHandle' en un vector y llama a '.join()'.\n");

    println!("✅ Todos los casos de error comprendidos.");
}
