//! ============================================================================
//! 🦀 FASE 2 - CLASE 03: Memoria Compartida con `Arc<T>` y `Mutex<T>` en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Qué es `Arc<T>`: Atomic Reference Counting para compartir propiedad entre hilos.
//! 2. Qué es `Mutex<T>`: Exclusión Mutua con liberación automática RAII (`MutexGuard`).
//! 3. El patrón `Arc<Mutex<T>>`: la llave universal para datos mutables compartidos.
//! 4. Por qué `Rc<T>` está prohibido entre hilos por el compilador de Rust.
//! 5. Caso real en IA: Entrenamiento paralelo donde 4 hilos actualizan los mismos pesos.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c03_teoria

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Representa los parámetros globales de un modelo de IA en memoria compartida.
#[derive(Debug)]
struct ParametrosModelo {
    peso: f32,
    sesgo: f32,
    pasos_entrenamiento: usize,
}

fn main() {
    println!("========================================================");
    println!("🔐 FASE 2 - CLASE 03: Memoria Compartida (Arc<Mutex<T>>)");
    println!("========================================================\n");

    // =========================================================================
    // 1. CREACIÓN DEL MODELO EN MEMORIA COMPARTIDA
    // =========================================================================
    println!("--- 1. Inicializar Modelo Protegido por Arc<Mutex<T>> ---");

    // 1. Mutex: Protege los datos con un cerrojo exclusivo.
    // 2. Arc: Permite que múltiples hilos sean co-propietarios del mismo puntero en Heap.
    let modelo_compartido = Arc::new(Mutex::new(ParametrosModelo {
        peso: 0.0,
        sesgo: 0.0,
        pasos_entrenamiento: 0,
    }));

    println!("Modelo inicial en RAM: {:?}", modelo_compartido.lock().unwrap());

    // =========================================================================
    // 2. PARALELISMO DE DATOS: 4 HILOS ENTRENANDO EN PARALELO
    // =========================================================================
    println!("\n--- 2. Lanzando 4 Hilos de Entrenamiento Paralelo ---");

    // Supongamos 4 lotes independientes procesados por 4 núcleos del CPU:
    let lotes_gradientes = [
        (0.50, 0.10),  // (grad_peso, grad_sesgo) calculado por Hilo 1
        (-0.20, 0.05), // calculado por Hilo 2
        (0.80, -0.15), // calculado por Hilo 3
        (0.10, 0.02),  // calculado por Hilo 4
    ];

    let mut handles = Vec::new();
    let learning_rate = 0.05;

    for (hilo_id, (grad_w, grad_b)) in lotes_gradientes.into_iter().enumerate() {
        // 💡 CLAVE: Clonamos el Arc (incrementa el contador atómico en 1, NO duplica el modelo):
        let modelo_clon = Arc::clone(&modelo_compartido);

        let h = thread::spawn(move || {
            // FASE A: Simula el Backpropagation independiente (cero locks):
            thread::sleep(Duration::from_millis(20 * (hilo_id + 1) as u64));

            // FASE B: Actualización de pesos (pedimos la llave del Mutex):
            // '.lock().unwrap()' bloquea hasta obtener acceso exclusivo:
            let mut modelo_acceso = modelo_clon.lock().unwrap();

            // Modificamos los pesos globales directamente en memoria:
            modelo_acceso.peso -= learning_rate * grad_w;
            modelo_acceso.sesgo -= learning_rate * grad_b;
            modelo_acceso.pasos_entrenamiento += 1;

            println!(
                "  [Hilo #{hilo_id}] Gradientes aplicados. Estado actual -> Peso: {:.4}, Sesgo: {:.4}",
                modelo_acceso.peso, modelo_acceso.sesgo
            );
            // 💡 El cerrojo se libera AUTOMÁTICAMENTE aquí cuando 'modelo_acceso' sale de scope!
        });

        handles.push(h);
    }

    // Esperamos a que todos los hilos terminen sus épocas:
    for h in handles {
        h.join().unwrap();
    }

    // =========================================================================
    // 3. RESULTADO FINAL TRAS LA FUSIÓN PARALELA
    // =========================================================================
    println!("\n--- 3. Parámetros Finales del Modelo ---");
    let modelo_final = modelo_compartido.lock().unwrap();
    println!("Pasos totales de optimización : {}", modelo_final.pasos_entrenamiento);
    println!("Peso global convergido        : {:.4}", modelo_final.peso);
    println!("Sesgo global convergido       : {:.4}", modelo_final.sesgo);

    println!("\n✅ Fin de la Clase 03 de Sistemas y Concurrencia.");
}
