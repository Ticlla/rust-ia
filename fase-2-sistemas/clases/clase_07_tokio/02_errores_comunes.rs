//! ============================================================================
//! 🛠️ FASE 2 - CLASE 07: Laboratorio de Errores Comunes con Tokio
//! ============================================================================
//!
//! Este laboratorio explora los 3 errores más críticos de producción al construir
//! servidores asíncronos de IA con Tokio.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c07_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Tokio en Servidores de IA");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Bloquear el hilo de Tokio con cálculo pesado de tensores
    // =========================================================================
    println!("--- Error 1: Cómputo CPU pesado directamente dentro de async ---");
    println!("Si dentro de un 'async fn' haces:");
    println!("    async fn inferencia_mala(tensor: Vec<f32>) {{");
    println!("        for _ in 0..10_000_000 {{ /* cálculo matricial pesado */ }}");
    println!("    }}");
    println!("\n¿Por qué es un desastre en producción?");
    println!("Porque Tokio asume que tus tareas son COOPERATIVAS (ceden el turno con .await).");
    println!("Si un hilo de Tokio pasa 500 ms calculando sin hacer ningún .await,");
    println!("TODAS las peticiones de red asignadas a ese hilo se congelan.");
    println!("👉 SOLUCIÓN: Envuelve siempre el cómputo pesado de tensores con:");
    println!("    tokio::task::spawn_blocking(move || {{ /* cómputo matricial */ }}).await\n");

    // =========================================================================
    // ERROR COMÚN 2: Guardar std::sync::MutexGuard a través de un .await
    // =========================================================================
    println!("--- Error 2: Mantener std::sync::Mutex a través de un .await ---");
    println!("Si intentas escribir:");
    println!("    let mut guard = modelo.lock().unwrap(); // std::sync::Mutex");
    println!("    descargar_datos_de_red().await;         // ❌ ERROR DE COMPILACIÓN");
    println!("    guard.usar();");
    println!("\n¿Por qué falla el compilador?");
    println!("Porque una tarea asíncrona puede pausarse en el Hilo 1 del CPU y");
    println!("reanudarse más tarde en el Hilo 2. El cerrojo 'std::sync::MutexGuard' está");
    println!("atado al ID del hilo del sistema operativo que lo creó.");
    println!("👉 MENSAJE: '`std::sync::MutexGuard` cannot be sent between threads safely'");
    println!("👉 SOLUCIÓN A: Suelta el cerrojo antes del .await (limitando el scope {{ }}).");
    println!("👉 SOLUCIÓN B: Usar 'tokio::sync::Mutex' si necesitas retenerlo entre awaits.\n");

    // =========================================================================
    // ERROR COMÚN 3: Tareas de fondo que mueren en silencio
    // =========================================================================
    println!("--- Error 3: 'Disparar y olvidar' (Fire and Forget) sin JoinHandle ---");
    println!("Si haces:");
    println!("    tokio::spawn(async {{ guardar_checkpoint_en_disco().await; }});");
    println!("    // y sales de main() de inmediato...");
    println!("\n¿Qué pasa?");
    println!("Cuando la función principal termina, TODO el runtime de Tokio se apaga.");
    println!("Tu tarea de guardado de checkpoint muere a la mitad, corrompiendo el archivo.");
    println!("👉 SOLUCIÓN: Guarda el 'JoinHandle' y hazle '.await' antes de cerrar el servidor.\n");

    println!("✅ Todos los casos de error comprendidos.");
}
