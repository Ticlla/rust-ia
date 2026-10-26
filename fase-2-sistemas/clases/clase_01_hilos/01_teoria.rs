//! ============================================================================
//! 🦀 FASE 2 - CLASE 01: Hilos Nativos (Threads), `join()` y `move` en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Creación de hilos del Sistema Operativo con `std::thread::spawn`.
//! 2. Esperar a que un hilo termine y obtener su resultado con `handle.join()`.
//! 3. El closure `move`: por qué los hilos deben ser dueños de sus datos.
//! 4. Diferencia brutal con Python (cero GIL: paralelismo real de núcleos CPU).
//! 5. Caso real en IA: Procesar 4 batches de datos en 4 hilos paralelos independientes.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c01_teoria

use std::thread;
use std::time::Duration;

fn main() {
    println!("========================================================");
    println!("⚡ FASE 2 - CLASE 01: Hilos Nativos (Threads) para IA");
    println!("========================================================\n");

    // =========================================================================
    // 1. CREACIÓN BÁSICA DE UN HILO PARALELO
    // =========================================================================
    println!("--- 1. Lanzar un hilo en segundo plano ---");

    // 'thread::spawn' recibe un closure y arranca un hilo nativo del SO en otro núcleo
    let handle = thread::spawn(|| {
        println!("  [Hilo Secundario] Hola desde el núcleo del CPU!");
        thread::sleep(Duration::from_millis(50)); // Simula un cómputo pesado
        42 // ¡Los hilos en Rust pueden RETORNAR valores directamente!
    });

    println!("  [Hilo Principal] El hilo secundario está corriendo en paralelo...");

    // Con '.join()' bloqueamos el hilo principal hasta que el secundario termine,
    // y obtenemos su valor de retorno envuelto en un Result:
    let resultado = handle.join().expect("El hilo falló (panic)");
    println!("  [Hilo Principal] Hilo terminado. Resultado recibido: {resultado}\n");

    // =========================================================================
    // 2. EL CLOSURE 'move': TRANSFERENCIA DE PROPIEDAD A HILOS
    // =========================================================================
    // ¿Por qué los hilos exigen 'move'?
    // Porque Rust no sabe cuánto tiempo vivirá el hilo secundario.
    // Si solo le prestáramos una referencia (&), 'main' podría terminar primero
    // y destruir los datos, dejando al hilo secundario con un puntero colgante (segfault).
    println!("--- 2. Transferencia de Propiedad con 'move' ---");

    let batch_datos = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // 'move' obliga al closure a ADUEÑARSE de 'batch_datos':
    let worker = thread::spawn(move || {
        let suma: f32 = batch_datos.iter().sum();
        let media = suma / batch_datos.len() as f32;
        media
    });

    // 'batch_datos' ya no existe en main porque fue movido al hilo.
    let media_calculada = worker.join().unwrap();
    println!("  Media calculada en el hilo secundario: {media_calculada}\n");

    // =========================================================================
    // 3. CASO REAL EN IA: PROCESAMIENTO PARALELO MULTI-HILO DE BATCHES
    // =========================================================================
    // Supongamos que tenemos 4 lotes de datos independientes y queremos
    // calcular la pérdida MSE de cada uno en paralelo usando 4 hilos:
    println!("--- 3. Pipeline Paralelo: 4 Batches en 4 Hilos de CPU ---");

    let batches = vec![
        vec![0.85, 0.90, 0.92],
        vec![0.40, 0.45, 0.50],
        vec![0.10, 0.15, 0.20],
        vec![0.02, 0.05, 0.08],
    ];

    let mut handles = Vec::new();

    for (batch_id, lote) in batches.into_iter().enumerate() {
        // Lanzamos un hilo por cada lote:
        let h = thread::spawn(move || {
            // Simulamos el cómputo de inferencia:
            let loss_promedio: f32 = lote.iter().sum::<f32>() / lote.len() as f32;
            (batch_id, loss_promedio) // Retornamos una tupla (id, resultado)
        });
        handles.push(h);
    }

    // Recolectamos los resultados a medida que cada hilo va terminando:
    for h in handles {
        let (id, loss) = h.join().unwrap();
        println!("  -> Batch #{id} terminado en paralelo con Loss = {loss:.4}");
    }

    println!("\n✅ Fin de la Clase 01 de Sistemas y Concurrencia.");
}
