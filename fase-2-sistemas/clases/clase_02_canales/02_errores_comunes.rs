//! ============================================================================
//! 🛠️ FASE 2 - CLASE 02: Laboratorio de Errores Comunes con Canales (`mpsc`)
//! ============================================================================
//!
//! Este laboratorio te enseña los 3 errores clásicos al trabajar con canales
//! en sistemas concurrentes y pipelines de datos en Rust.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c02_errores

fn main() {
    println!("========================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Canales mpsc en Rust");
    println!("========================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Usar un tensor después de enviarlo por el canal
    // =========================================================================
    println!("--- Error 1: Usar una variable después de tx.send() ---");
    println!("Si intentas escribir:");
    println!("    let buffer = vec![1.0, 2.0, 3.0];");
    println!("    tx.send(buffer).unwrap();");
    println!("    println!(\"{{:?}}\", buffer); // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque 'send()' toma posesión (Ownership) del buffer y lo transfiere");
    println!("a través del canal hacia el otro hilo. En el hilo emisor ya no existe.");
    println!("👉 MENSAJE DE RUST: 'use of moved value: buffer'");
    println!("👉 VENTAJA: Cero riesgo de que dos hilos lean o modifiquen el mismo tensor a la vez.\n");

    // =========================================================================
    // ERROR COMÚN 2: El bloqueo infinito (Olvidar 'drop(tx)' en main)
    // =========================================================================
    println!("--- Error 2: El Deadlock por olvidar drop(tx) ---");
    println!("Cuando tienes varios workers y haces:");
    println!("    let tx1 = tx.clone();");
    println!("    let tx2 = tx.clone();");
    println!("    for lote in rx {{ ... }} // ❌ SE QUEDA CONGELADO PARA SIEMPRE\n");
    println!("¿Por qué ocurre?");
    println!("Porque el receptor 'rx' solo sabe que los envíos terminaron cuando");
    println!("ABSOLUTAMENTE TODOS los transmisores han muerto.");
    println!("Como la variable original 'tx' en main sigue viva, 'rx' se queda esperando");
    println!("eternamente pensando que main podría enviar otro mensaje.");
    println!("👉 SOLUCIÓN: Escribir 'drop(tx);' en main antes de empezar a consumir de 'rx'.\n");

    // =========================================================================
    // ERROR COMÚN 3: Intentar clonar el Receptor (rx.clone())
    // =========================================================================
    println!("--- Error 3: Intentar clonar 'rx' (MPSC = Single Consumer) ---");
    println!("Si intentas escribir:");
    println!("    let rx2 = rx.clone(); // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque MPSC significa 'Multiple Producer, SINGLE Consumer'.");
    println!("El canal garantiza que solo UNA entidad puede extraer los mensajes.");
    println!("👉 Si necesitas que varios consumidores lean del mismo canal, se usan");
    println!("   canales de difusión (broadcast) que veremos en clases avanzadas.\n");

    println!("✅ Todos los casos de error comprendidos.");
}
