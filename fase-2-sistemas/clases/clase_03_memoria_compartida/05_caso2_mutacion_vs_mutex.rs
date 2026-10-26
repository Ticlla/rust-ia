//! ============================================================================
//! 🧪 CASO 2: Mutación Concurrente (&mut) vs Exclusión Mutua (Mutex)
//! ============================================================================
//!
//! Ejecutar este caso:
//!     cargo run --bin c03_caso2
//!
//! CONTEXTO EN IA:
//! Dos hilos calculan gradientes en lotes distintos y quieren aplicar el paso de
//! optimización: `peso -= lr * gradiente` sobre los mismos parámetros.

use std::sync::Mutex;
use std::thread;
use std::time::Duration;

fn main() {
    println!("============================================================");
    println!("🧪 CASO 2: 'cannot borrow as mutable twice' [E0499] vs Mutex");
    println!("============================================================\n");

    // =========================================================================
    // PARTE A: ¿CUÁL ES EL ERROR DE NOVATO?
    // =========================================================================
    println!("--- ❌ PARTE A: El Error que cometerías en C++ ---");
    println!("Si intentas escribir:");
    println!("    let mut pesos = vec![1.0, 2.0, 3.0];");
    println!("    thread::scope(|s| {{");
    println!("        s.spawn(|| pesos[0] += 0.5);");
    println!("        s.spawn(|| pesos[0] -= 0.2); // ❌ ERROR!\n");
    println!("    }});");

    println!("¿Qué dice el compilador de Rust?");
    println!("    error[E0499]: cannot borrow `pesos` as mutable more than once at a time");
    println!("    first mutable borrow occurs here (Hilo 1)");
    println!("    second mutable borrow occurs here (Hilo 2)\n");

    println!("¿Por qué falla?");
    println!("Porque dos hilos escribiendo al mismo tiempo en el mismo byte causan");
    println!("una Carrera de Datos (Data Race), mutilando los bits del número decimal (f32).\n");

    // =========================================================================
    // PARTE B: LA SOLUCIÓN CORRECTA CON Mutex<T>
    // =========================================================================
    println!("--- ✅ PARTE B: La Solución con Mutex<T> (Exclusión Mutua Segura) ---");

    // Envolvemos el modelo dentro del cerrojo:
    let pesos_protegidos = Mutex::new(vec![1.0, 2.0, 3.0]);
    println!("1. Pesos iniciales dentro del Mutex: {:?}", pesos_protegidos.lock().unwrap());

    let lr = 0.1;

    // Usamos thread::scope para sincronizar los hilos:
    thread::scope(|s| {
        // Hilo 1: calcula gradiente y actualiza
        s.spawn(|| {
            println!("  [Hilo 1] Calculando gradiente en lote A...");
            thread::sleep(Duration::from_millis(30));
            let gradiente_a = 5.0; // Gradiente simulado

            println!("  [Hilo 1] Solicitando llave del Mutex...");
            let mut pesos = pesos_protegidos.lock().unwrap();
            println!("  [Hilo 1] 🔒 Llave adquirida! Aplicando actualización...");
            pesos[0] -= lr * gradiente_a;
            thread::sleep(Duration::from_millis(20)); // Tiempo de escritura
            println!("  [Hilo 1] 🔓 Actualización lista. Soltando llave!");
            // La llave se libera automáticamente al terminar la función del hilo (RAII)
        });

        // Hilo 2: calcula gradiente y actualiza
        s.spawn(|| {
            println!("  [Hilo 2] Calculando gradiente en lote B...");
            thread::sleep(Duration::from_millis(40));
            let gradiente_b = -2.0; // Gradiente simulado

            println!("  [Hilo 2] Solicitando llave del Mutex...");
            let mut pesos = pesos_protegidos.lock().unwrap();
            println!("  [Hilo 2] 🔒 Llave adquirida! Aplicando actualización...");
            pesos[0] -= lr * gradiente_b;
            thread::sleep(Duration::from_millis(20));
            println!("  [Hilo 2] 🔓 Actualización lista. Soltando llave!");
        });
    });

    println!("\n2. Ambos hilos terminaron sin colisiones ni corrupción de memoria.");
    let pesos_finales = pesos_protegidos.lock().unwrap();
    println!("Pesos finales sincronizados: {:?}", *pesos_finales);
    println!("Esperado para peso[0]: 1.0 - (0.1 * 5.0) - (0.1 * -2.0) = 0.7");
    println!("\n✅ Caso 2 completado con éxito.");
}

// =============================================================================
// CÓDIGO CON ERROR PARA DESCOMENTAR Y COMPROBAR:
// =============================================================================
// Si quieres ver al compilador fallar en vivo, descomenta esta función:
//
// fn codigo_invalido() {
//     let mut pesos = vec![1.0, 2.0, 3.0];
//     thread::scope(|s| {
//         s.spawn(|| pesos[0] += 0.5);
//         s.spawn(|| pesos[0] -= 0.2); // ❌ error[E0499]
//     });
// }
