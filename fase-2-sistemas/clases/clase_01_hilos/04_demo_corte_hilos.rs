//! ============================================================================
//! 🧪 EXPERIMENTO EN VIVO: ¿Qué pasa cuando 'main' termina antes que un hilo?
//! ============================================================================
//!
//! Ejecutar este experimento:
//!     cargo run --bin c01_demo

use std::thread;
use std::time::Duration;

fn main() {
    println!("========================================================");
    println!("🧪 EXPERIMENTO: Hilo cortado vs. Hilo esperado con join()");
    println!("========================================================\n");

    // =========================================================================
    // EXPERIMENTO 1: SIN .join() (El hilo secundario es asesinado a mitad)
    // =========================================================================
    println!("--- EXPERIMENTO 1: Sin .join() ---");
    println!("Vamos a lanzar un hilo que quiere contar del 1 al 5...");

    thread::spawn(|| {
        for i in 1..=5 {
            println!("  [Hilo Secundario] Procesando batch #{i}...");
            thread::sleep(Duration::from_millis(100)); // Cómputo más pesado
        }
        println!("  [Hilo Secundario] 🏁 ¡Terminé todo mi trabajo!");
    });

    // El hilo principal hace una tarea brevísima y termina de inmediato:
    println!("[Hilo Principal] 🚪 Me voy inmediatamente, no espero a nadie.");
    // ⚠️ AL TERMINAR MAIN, EL HILO SECUNDARIO ES DESTRUIDO AL INSTANTE.
    // ⚠️ AL LLEGAR AQUÍ, LINUX DESTRUYE EL PROCESO COMPLETO.
    // Observa en la terminal cómo el hilo secundario NUNCA llega al batch #5 ni a la bandera.
    
    // Pausa dramática para que veas la diferencia:
    thread::sleep(Duration::from_millis(300));
    println!("\n--------------------------------------------------------");

    // =========================================================================
    // EXPERIMENTO 2: CON .join() (El hilo principal espera educadamente)
    // =========================================================================
    println!("--- EXPERIMENTO 2: Con .join() ---");
    println!("Ahora guardamos el 'handle' y le pedimos a main que espere...");

    // Guardamos el control del hilo en la variable 'handle':
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("  [Hilo Secundario B] Procesando batch #{i}...");
            thread::sleep(Duration::from_millis(30));
        }
        println!("  [Hilo Secundario B] 🏁 ¡Trabajo completado al 100%!");
        "Resultado del modelo: 99.8% de precisión" // Valor que retorna el hilo
    });

    println!("[Hilo Principal] Esperando a que el hilo secundario termine...");

    // ¡La magia de join()!
    // Bloquea aquí hasta que el hilo termine y nos entrega su resultado:
    let mensaje_final = handle.join().unwrap();

    println!("[Hilo Principal] El hilo secundario terminó en orden.");
    println!("[Hilo Principal] Recibido: \"{mensaje_final}\"");

    println!("\n✅ Fin del experimento.");
}
