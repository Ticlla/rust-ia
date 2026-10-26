//! ============================================================================
//! 🦀 CLASE 01: TEORÍA Y EJEMPLOS - Variables y Mutabilidad en Rust para IA
//! ============================================================================
//!
//! Ejecutar con:
//!     cargo run --bin c01_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 01 [TEORÍA]: Variables y Mutabilidad para IA");
    println!("========================================================\n");

    // -------------------------------------------------------------------------
    // 1. INMUTABILIDAD POR DEFECTO
    // -------------------------------------------------------------------------
    println!("--- 1. Inmutabilidad por Defecto ---");
    let learning_rate: f32 = 0.01;
    println!("Learning rate inicial (inmutable): {}", learning_rate);
    // En Rust, si intentas reasignar: learning_rate = 0.005; el compilador lo prohíbe.
    // learning_rate = 0.02; error[E0384]: cannot assign twice to immutable variable `learning_rate`

    // -------------------------------------------------------------------------
    // 2. MUTABILIDAD EXPLÍCITA CON `let mut`
    // -------------------------------------------------------------------------
    println!("\n--- 2. Mutabilidad Explícita con 'mut' ---");
    let mut epoca: usize = 0;
    let mut loss_acumulada: f32 = 1.450;

    println!("Estado inicial -> Época: {}, Pérdida: {:.3}", epoca, loss_acumulada);

    // Simulamos un paso de entrenamiento:
    epoca += 1;
    loss_acumulada -= 0.320;
    println!("Paso 1 completado -> Época: {}, Pérdida: {:.3}", epoca, loss_acumulada);

    // -------------------------------------------------------------------------
    // 3. CONSTANTES (`const`) vs VARIABLES INMUTABLES (`let`)
    // -------------------------------------------------------------------------
    println!("\n--- 3. Constantes de Tiempo de Compilación ---");
    const MAX_EPOCAS: usize = 500;
    const GAMMA: f32 = 0.99; // Factor de descuento en RL
    const MODEL_NAME: &str = "ResNet-50-IA";

    println!("Modelo       : {}", MODEL_NAME);
    println!("Max Épocas   : {}", MAX_EPOCAS);
    println!("Gamma (RL)   : {}", GAMMA);

    // -------------------------------------------------------------------------
    // 4. SHADOWING (SOMBREADO) EN PIPELINES DE DATOS
    // -------------------------------------------------------------------------
    println!("\n--- 4. Shadowing en Preprocesamiento de Datos ---");

    // Recibimos la dimensión de batch desde una configuración de usuario (texto)
    let batch_size = "128";
    println!("  Paso 1 (Texto)     : \"{}\"", batch_size);

    // Transformamos a número usando el mismo nombre:
    let batch_size: usize = batch_size.parse().expect("Debe ser un número válido");
    println!("  Paso 2 (Numérico)  : {}", batch_size);

    // Ajustamos para entrenamiento distribuido con 4 GPUs:
    let batch_size = batch_size * 4;
    println!("  Paso 3 (Multi-GPU) : {} muestras por paso global", batch_size);

    println!("\n✅ Ejecución de teoría completada con éxito.");
    println!("👉 Prueba ahora el laboratorio de errores con: cargo run --bin c01_errores");
}
