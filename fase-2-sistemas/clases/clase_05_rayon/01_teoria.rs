//! ============================================================================
//! 🦀 FASE 2 - CLASE 05: Paralelismo de Datos con `Rayon` en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Paralelismo de Tareas (`thread::spawn`) vs Paralelismo de Datos (`Rayon`).
//! 2. El algoritmo de "Work Stealing" (Robo de Trabajo) que usa Rayon.
//! 3. De `iter()` a `par_iter()`: Paralelizar operaciones matriciales con 1 palabra.
//! 4. Caso real en IA: Activación GELU y normalización de 5 millones de parámetros.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c05_teoria

use rayon::prelude::*;
use std::time::Instant;

/// Función de activación GELU (Gaussian Error Linear Unit), estándar en Transformers/LLMs.
/// Aproximación matemática rápida: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978846;
    let factor = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
    0.5 * x * (1.0 + factor.tanh())
}

fn main() {
    println!("============================================================");
    println!("⚡ FASE 2 - CLASE 05: Paralelismo de Datos con Rayon (par_iter)");
    println!("============================================================\n");

    let num_nucleos = rayon::current_num_threads();
    println!("💻 Rayon detectó {} núcleos lógicos disponibles en tu CPU.\n", num_nucleos);

    // =========================================================================
    // 1. CREACIÓN DE UN TENSOR GRANDE (5 MILLONES DE FLOATS)
    // =========================================================================
    let n = 5_000_000;
    println!("--- 1. Generando tensor sintético de {n} elementos (f32) ---");

    // Llenamos el tensor con valores en el rango [-3.0, 3.0]:
    let mut tensor_secuencial: Vec<f32> = (0..n)
        .map(|i| (i as f32 % 600.0 - 300.0) / 100.0)
        .collect();

    let mut tensor_paralelo = tensor_secuencial.clone();

    // =========================================================================
    // 2. COMPUTO SECUENCIAL TRADICIONAL (1 SOLO NÚCLEO)
    // =========================================================================
    println!("\n--- 2. Ejecutando Activación GELU SECUENCIAL (1 núcleo CPU) ---");
    let inicio_sec = Instant::now();

    // Bucle secuencial clásico (.iter_mut()):
    tensor_secuencial.iter_mut().for_each(|x| {
        *x = gelu(*x);
    });

    let duracion_sec = inicio_sec.elapsed();
    println!("⏱️  Tiempo Secuencial : {:.2?}", duracion_sec);

    // =========================================================================
    // 3. COMPUTO PARALELO CON RAYON (TODOS LOS NÚCLEOS DEL CPU)
    // =========================================================================
    println!("\n--- 3. Ejecutando Activación GELU PARALELA con Rayon (.par_iter_mut()) ---");
    let inicio_par = Instant::now();

    // 💡 LA MAGIA DE RAYON: Reemplazamos '.iter_mut()' por '.par_iter_mut()'
    // Rayon divide el vector en fragmentos óptimos y los reparte en el Thread Pool:
    tensor_paralelo.par_iter_mut().for_each(|x| {
        *x = gelu(*x);
    });

    let duracion_par = inicio_par.elapsed();
    println!("⏱️  Tiempo Paralelo   : {:.2?}", duracion_par);

    // Verificamos que ambos resultados sean idénticos:
    let son_iguales = tensor_secuencial
        .iter()
        .zip(tensor_paralelo.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);

    assert!(son_iguales, "Los resultados de Rayon difieren del secuencial!");
    println!("✅ Precisión matemática verificada: los resultados son idénticos.");

    if duracion_par.as_nanos() > 0 {
        let speedup = duracion_sec.as_secs_f64() / duracion_par.as_secs_f64();
        println!("🚀 Aceleración (Speedup) : {:.2}x más rápido", speedup);
    }

    // =========================================================================
    // 4. MAP-REDUCE PARALELO EN IA: NORMA L2 DEL TENSOR
    // =========================================================================
    println!("\n--- 4. Map-Reduce Paralelo: Cálculo de la Norma L2 (||X||) ---");
    let inicio_l2 = Instant::now();

    // Suma paralela de cuadrados usando .par_iter().map().sum():
    let suma_cuadrados: f32 = tensor_paralelo
        .par_iter()
        .map(|&x| x * x)
        .sum();

    let norma_l2 = suma_cuadrados.sqrt();
    println!("Norma L2 calculada en paralelo: {:.4} (Tiempo: {:.2?})", norma_l2, inicio_l2.elapsed());

    println!("\n✅ Fin de la demostración teórica de Rayon.");
}
