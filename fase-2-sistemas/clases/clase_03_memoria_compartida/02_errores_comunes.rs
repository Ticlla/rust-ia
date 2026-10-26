//! ============================================================================
//! 🛠️ FASE 2 - CLASE 03: Laboratorio de Errores con `Arc` y `Mutex`
//! ============================================================================
//!
//! Este laboratorio te enseña los 3 errores críticos de concurrencia al compartir
//! memoria entre hilos y cómo el compilador de Rust los previene.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c03_errores

fn main() {
    println!("========================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Arc y Mutex en Rust");
    println!("========================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Intentar usar Rc<T> en lugar de Arc<T> entre hilos
    // =========================================================================
    println!("--- Error 1: Usar Rc<T> en lugar de Arc<T> entre hilos ---");
    println!("Si intentas escribir:");
    println!("    use std::rc::Rc;");
    println!("    let datos = Rc::new(vec![1.0, 2.0]);");
    println!("    thread::spawn(move || println!(\"{{:?}}\", datos)); // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque 'Rc' (Reference Counting normal) incrementa su contador con operaciones");
    println!("normales de CPU no sincronizadas. Si dos hilos incrementaran 'Rc' a la vez,");
    println!("el contador se corrompería.");
    println!("👉 MENSAJE DE RUST: '`Rc<Vec<f32>>` cannot be sent between threads safely'");
    println!("👉 SOLUCIÓN: Usar 'Arc<T>' (A = Atomic), que usa instrucciones atómicas del CPU.\n");

    // =========================================================================
    // ERROR COMÚN 2: Intentar modificar dentro de Arc<T> sin un Mutex
    // =========================================================================
    println!("--- Error 2: Intentar mutar dentro de Arc sin cerrojo ---");
    println!("Si intentas escribir:");
    println!("    let pesos = Arc::new(vec![0.5, 0.8]);");
    println!("    pesos[0] = 0.9; // ❌ ERROR\n");
    println!("¿Por qué falla el compilador?");
    println!("Porque 'Arc<T>' por sí solo solo otorga referencias de SOLO LECTURA (&T).");
    println!("Para permitir mutabilidad segura entre varios hilos, debes envolverlo con Mutex:");
    println!("👉 SOLUCIÓN: 'Arc<Mutex<T>>'.\n");

    // =========================================================================
    // ERROR COMÚN 3: El Deadlock (Bloqueo Mortal / Doble Lock)
    // =========================================================================
    println!("--- Error 3: El Deadlock por pedir el mismo lock dos veces ---");
    println!("Si en el mismo hilo escribes:");
    println!("    let cerrojo = Mutex::new(10);");
    println!("    let g1 = cerrojo.lock().unwrap();");
    println!("    let g2 = cerrojo.lock().unwrap(); // 💀 DEADLOCK: Se congela para siempre\n");
    println!("¿Por qué ocurre?");
    println!("Porque la línea 2 ya tiene la llave. La línea 3 se sienta a esperar");
    println!("a que la línea 2 suelte la llave... ¡pero la línea 2 nunca la soltará");
    println!("porque el hilo está congelado en la línea 3!");
    println!("👉 SOLUCIÓN: Limita el alcance ({{ }}) de los locks para que se liberen rápido.\n");

    println!("✅ Todos los casos de error comprendidos.");
}
