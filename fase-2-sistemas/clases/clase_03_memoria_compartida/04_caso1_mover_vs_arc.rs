//! ============================================================================
//! 🧪 CASO 1: Mover la Propiedad (move) vs Co-propiedad Atómica (Arc)
//! ============================================================================
//!
//! Ejecutar este caso:
//!     cargo run --bin c03_caso1
//!
//! CONTEXTO EN IA:
//! Queremos que dos hilos en segundo plano lean los mismos pesos de una red neuronal
//! en paralelo (por ejemplo: el Hilo 1 evalúa el set de Validación y el Hilo 2
//! evalúa el set de Test).

use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn main() {
    println!("============================================================");
    println!("🧪 CASO 1: 'use of moved value' [E0382] vs Solución con Arc");
    println!("============================================================\n");

    // =========================================================================
    // PARTE A: ¿CUÁL ES EL ERROR DE NOVATO?
    // =========================================================================
    println!("--- ❌ PARTE A: El Error que cometerías en C++ o Python ---");
    println!("Si intentas escribir:");
    println!("    let pesos = vec![0.5, 0.8, -0.2];");
    println!("    thread::spawn(move || println!(\"Hilo 1: {{:?}}\", pesos));");
    println!("    thread::spawn(move || println!(\"Hilo 2: {{:?}}\", pesos)); // ❌ ERROR!\n");

    println!("¿Qué dice el compilador de Rust?");
    println!("    error[E0382]: use of moved value: `pesos`");
    println!("    value moved into closure here, in previous iteration\n");

    println!("¿Por qué falla?");
    println!("Porque la palabra clave 'move' transfiere el Ownership exclusivo.");
    println!("El Hilo 1 se quedó con la propiedad del vector en el Heap.");
    println!("El Hilo 2 intenta usar una variable que ya no le pertenece al hilo principal.\n");

    // =========================================================================
    // PARTE B: LA SOLUCIÓN CORRECTA CON Arc<T>
    // =========================================================================
    println!("--- ✅ PARTE B: La Solución con Arc<T> (Cero copias de datos) ---");

    let pesos_modelo = vec![0.5, 0.8, -0.2];
    println!("1. Vector de pesos original en RAM: {:?}", pesos_modelo);

    // 💡 Envolvemos los pesos en un Arc:
    let modelo_compartido = Arc::new(pesos_modelo);
    println!(
        "2. Arc inicial creado. Contador de referencias: {}",
        Arc::strong_count(&modelo_compartido)
    );

    // Clonamos el Arc para el Hilo 1 (solo incrementa el contador en el CPU a 2):
    let modelo_hilo1 = Arc::clone(&modelo_compartido);
    println!(
        "3. Clon para Hilo 1. Contador de referencias: {}",
        Arc::strong_count(&modelo_compartido)
    );

    // Clonamos el Arc para el Hilo 2 (incrementa el contador a 3):
    let modelo_hilo2 = Arc::clone(&modelo_compartido);
    println!(
        "4. Clon para Hilo 2. Contador de referencias: {}",
        Arc::strong_count(&modelo_compartido)
    );

    let h1 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        println!(
            "  -> [Hilo 1 - Validación] Leyendo pesos en dirección {:p}: {:?}",
            modelo_hilo1.as_ptr(),
            *modelo_hilo1
        );
    });

    let h2 = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50));
        println!(
            "  -> [Hilo 2 - Test]       Leyendo pesos en dirección {:p}: {:?}",
            modelo_hilo2.as_ptr(),
            *modelo_hilo2
        );
    });

    h1.join().unwrap();
    h2.join().unwrap();

    println!(
        "\n5. Hilos terminados. Contador de referencias volvió a: {}",
        Arc::strong_count(&modelo_compartido)
    );
    println!("💡 Ambos hilos leyeron la EXACTA MISMA dirección de memoria física sin clonar los datos.");
    println!("\n✅ Caso 1 completado con éxito.");
}

// =============================================================================
// CÓDIGO CON ERROR PARA DESCOMENTAR Y COMPROBAR:
// =============================================================================
// Si quieres ver al compilador fallar en vivo, descomenta esta función:
//
// fn codigo_invalido() {
//     let pesos = vec![0.5, 0.8];
//     thread::spawn(move || println!("{:?}", pesos));
//     thread::spawn(move || println!("{:?}", pesos)); // ❌ error[E0382]
// }
