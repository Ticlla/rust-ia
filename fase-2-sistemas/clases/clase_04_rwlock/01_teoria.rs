//! ============================================================================
//! 🦀 FASE 2 - CLASE 04: Lecturas Concurrentes con `RwLock<T>` en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. El gran cuello de botella del `Mutex` en Inferencia de IA.
//! 2. Qué es `RwLock<T>` (Read-Write Lock) y cómo implementa "Muchos Lectores, Un Escritor".
//! 3. El patrón `Arc<RwLock<ModeloIA>>` para servidores de inferencia con actualización en caliente.
//! 4. Diferencia clave en rendimiento: Inferencia 100% paralela sin serialización.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c04_teoria

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Duration;

/// Modelo de IA con pesos que se usan para inferencia y se pueden actualizar.
#[derive(Debug)]
struct ModeloInferencia {
    version: usize,
    pesos: Vec<f32>,
}

impl ModeloInferencia {
    fn predecir(&self, entrada: &[f32]) -> f32 {
        // Simula inferencia: producto punto y = W * X
        self.pesos.iter().zip(entrada.iter()).map(|(w, x)| w * x).sum()
    }
}

fn main() {
    println!("============================================================");
    println!("📖 FASE 2 - CLASE 04: Servidor de Inferencia con RwLock<T>");
    println!("============================================================\n");

    // =========================================================================
    // 1. INICIALIZACIÓN DEL MODELO CON RwLock
    // =========================================================================
    println!("--- 1. Cargando Modelo Protegido por Arc<RwLock<T>> ---");

    let modelo_compartido = Arc::new(RwLock::new(ModeloInferencia {
        version: 1,
        pesos: vec![0.5, -0.2, 0.8],
    }));

    println!(
        "Modelo v1 listo en RAM: {:?}\n",
        modelo_compartido.read().unwrap()
    );

    // =========================================================================
    // 2. SIMULACIÓN: 4 CLIENTES DE INFERENCIA LEYENDO EN PARALELO
    // =========================================================================
    println!("--- 2. Lanzando 4 Hilos de Inferencia (Lectores Simultáneos) ---");

    let mut handles = Vec::new();

    // 4 hilos lectores (inferencia concurrente):
    for cliente_id in 0..4 {
        let modelo = Arc::clone(&modelo_compartido);
        let h = thread::spawn(move || {
            let entrada = vec![1.0, 2.0, 3.0];

            // 💡 '.read()' NO bloquea a otros lectores! Todos leen en paralelo.
            let modelo_guard = modelo.read().unwrap();
            println!(
                "  [Cliente #{cliente_id}] 🟢 Leyendo modelo v{} en paralelo...",
                modelo_guard.version
            );

            // Simula cómputo de inferencia:
            thread::sleep(Duration::from_millis(50));
            let pred = modelo_guard.predecir(&entrada);

            println!(
                "  [Cliente #{cliente_id}] 🎯 Predicción calculada: {:.4}",
                pred
            );
            // El cerrojo de lectura se libera automáticamente al salir de scope
        });
        handles.push(h);
    }

    // =========================================================================
    // 3. ACTUALIZACIÓN EN CALIENTE DEL MODELO (UN ESCRITOR EXCLUSIVO)
    // =========================================================================
    let modelo_para_actualizar = Arc::clone(&modelo_compartido);
    let h_escritor = thread::spawn(move || {
        // Esperamos a que los clientes arranquen:
        thread::sleep(Duration::from_millis(20));

        println!("\n  [Hilo Trainer] 🚨 Solicitando permiso de ESCRITURA (.write())...");
        // 💡 '.write()' esperará pacientemente a que los lectores activos terminen,
        // y bloqueará nuevas lecturas hasta que la actualización termine.
        let mut modelo_guard = modelo_para_actualizar.write().unwrap();
        println!("  [Hilo Trainer] 🔒 Llave de ESCRITURA adquirida. Actualizando pesos a v2...");

        modelo_guard.version = 2;
        modelo_guard.pesos = vec![1.0, 0.5, -0.3];
        thread::sleep(Duration::from_millis(60)); // Simula copiar nuevos pesos

        println!("  [Hilo Trainer] 🔓 Modelo actualizado a v2 con éxito!\n");
    });
    handles.push(h_escritor);

    // =========================================================================
    // 4. NUEVOS CLIENTES LLEGANDO DESPUÉS DE LA ACTUALIZACIÓN
    // =========================================================================
    for cliente_id in 4..6 {
        let modelo = Arc::clone(&modelo_compartido);
        let h = thread::spawn(move || {
            thread::sleep(Duration::from_millis(100)); // Llegan después de v2
            let entrada = vec![1.0, 2.0, 3.0];

            let modelo_guard = modelo.read().unwrap();
            println!(
                "  [Cliente #{cliente_id}] 🟢 Leyendo modelo v{} (¡nueva versión!)...",
                modelo_guard.version
            );
            let pred = modelo_guard.predecir(&entrada);
            println!(
                "  [Cliente #{cliente_id}] 🎯 Predicción con v2: {:.4}",
                pred
            );
        });
        handles.push(h);
    }

    // Esperamos a que todos terminen:
    for h in handles {
        h.join().unwrap();
    }

    // =========================================================================
    // 5. ESTADO FINAL
    // =========================================================================
    println!("\n--- 3. Estado Final del Servidor ---");
    let modelo_final = modelo_compartido.read().unwrap();
    println!("Versión activa en producción: v{}", modelo_final.version);
    println!("Pesos finales               : {:?}", modelo_final.pesos);
    println!("\n✅ Fin de la Clase 04 de Sistemas y Concurrencia.");
}
