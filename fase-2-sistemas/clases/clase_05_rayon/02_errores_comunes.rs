//! ============================================================================
//! 🛠️ FASE 2 - CLASE 05: Laboratorio de Errores Comunes con Rayon
//! ============================================================================
//!
//! Este laboratorio explora los 3 errores de rendimiento más comunes al aplicar
//! paralelismo de datos en IA.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c05_errores

use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Paralelismo de Datos con Rayon");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: Paralelizar datos diminutos (Overhead > Beneficio)
    // =========================================================================
    println!("--- Error 1: Paralelizar colecciones diminutas (Amdahl's Law) ---");
    let tensor_pequeno: Vec<f32> = (0..50).map(|x| x as f32).collect();

    let t0 = Instant::now();
    let _sum_sec: f32 = tensor_pequeno.iter().map(|x| x * 2.0).sum();
    let d_sec = t0.elapsed();

    let t1 = Instant::now();
    let _sum_par: f32 = tensor_pequeno.par_iter().map(|x| x * 2.0).sum();
    let d_par = t1.elapsed();

    println!("Vector de solo 50 elementos:");
    println!("  • Tiempo Secuencial : {:.2?}", d_sec);
    println!("  • Tiempo Rayon      : {:.2?}", d_par);
    println!("👉 LECCIÓN: Para datos pequeños, la coordinación entre hilos tarda más que la suma.");
    println!("👉 REGLA: Usa Rayon para tensores grandes (miles/millones de elementos) o cálculos intensivos.\n");

    // =========================================================================
    // ERROR COMÚN 2: Usar un Mutex dentro de par_iter (Contención Brutal)
    // =========================================================================
    println!("--- Error 2: Meter un Mutex dentro de un bucle par_iter ---");
    let datos: Vec<f32> = (0..100_000).map(|x| x as f32).collect();

    // ❌ FORMA INCORRECTA (Antipatrón):
    let acumulador_lento = Mutex::new(0.0);
    let t_mutex = Instant::now();
    datos.par_iter().for_each(|&x| {
        let mut g = acumulador_lento.lock().unwrap(); // ¡16 hilos peleándose por el cerrojo!
        *g += x;
    });
    let d_mutex = t_mutex.elapsed();

    // ✅ FORMA CORRECTA CON RAYON (Map-Reduce nativo):
    let t_reduce = Instant::now();
    let _suma_correcta: f32 = datos.par_iter().sum(); // Cero locks, suma paralela por ramas
    let d_reduce = t_reduce.elapsed();

    println!("Sumar 100,000 números en paralelo:");
    println!("  • Con Mutex dentro de par_iter : {:.2?}", d_mutex);
    println!("  • Con par_iter().sum()         : {:.2?}", d_reduce);
    println!("👉 LECCIÓN: Un Mutex dentro de par_iter destruye el paralelismo y crea cuellos de botella.");
    println!("👉 REGLA: En lugar de Mutex, usa 'sum()', 'reduce()', o 'fold()'.\n");

    // =========================================================================
    // ERROR COMÚN 3: Asumir que par_iter procesa en orden secuencial
    // =========================================================================
    println!("--- Error 3: Asumir orden secuencial en operaciones paralelas ---");
    let elementos = vec!["token_0", "token_1", "token_2", "token_3", "token_4"];

    print!("Procesamiento secuencial : ");
    elementos.iter().for_each(|t| print!("{t} "));
    println!();

    print!("Procesamiento con Rayon  : ");
    elementos.par_iter().for_each(|t| print!("{t} "));
    println!();
    println!("👉 LECCIÓN: 'par_iter' divide en trozos y 'roba trabajo'. El orden de llegada no es determinista.");
    println!("👉 REGLA: Si el orden importa (ej: tokens en un LLM), mantén índices o usa .collect() ordenado.\n");

    println!("✅ Todos los casos de error y antipatrones comprendidos.");
}
