//! ============================================================================
//! 🦀 CLASE 01: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! ============================================================================
//!
//! En este archivo pondrás a prueba lo aprendido en la Clase 01.
//! Completa los retos guiados y verifica tus respuestas ejecutando:
//!
//!     cargo run --bin c01_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 01 [EJERCICIOS PRÁCTICOS]: Variables y Mutabilidad");
    println!("========================================================\n");

    // -------------------------------------------------------------------------
    // RETO 1: Scheduler de Learning Rate (Decaimiento de tasa con 'mut')
    // -------------------------------------------------------------------------
    println!("--- Reto 1: Decaimiento del Learning Rate ---");
    // Contexto: Al entrenar una red neuronal, es común reducir la tasa de
    // aprendizaje a la mitad cada 10 épocas para asegurar convergencia fina.

    let mut lr: f32 = 0.1;
    let factor_decaimiento: f32 = 0.5;

    // Simulamos 3 reducciones:
    lr = lr * factor_decaimiento; // Paso 1
    lr = lr * factor_decaimiento; // Paso 2
    lr = lr * factor_decaimiento; // Paso 3

    println!("Tasa inicial: 0.1 | Tasa final tras 3 decaimientos: {:.4}", lr);
    assert!((lr - 0.0125).abs() < 1e-6, "El cálculo del LR debe ser 0.0125");
    println!("✅ Reto 1 superado!\n");

    // -------------------------------------------------------------------------
    // RETO 2: Pipeline de Preprocesamiento de Hiperparámetros con Shadowing
    // -------------------------------------------------------------------------
    println!("--- Reto 2: Pipeline de Shadowing ---");
    // Contexto: Un archivo de configuración devuelve la longitud máxima de
    // tokens de un LLM como texto con espacios (" 2048 ").
    // Debes:
    //   1. Recibirlo como &str.
    //   2. Limpiar espacios (.trim()) y parsearlo a usize.
    //   3. Duplicarlo para soportar context window extendido.

    let context_window = "  2048  ";
    println!("1. Valor en bruto (texto): \"{}\"", context_window);

    // Sombreado a tipo numérico usize:
    let context_window: usize = context_window.trim().parse().expect("Debe ser numérico");
    println!("2. Valor parseado (usize): {}", context_window);

    // Sombreado a ventana duplicada:
    let context_window = context_window * 2;
    println!("3. Ventana extendida final: {}", context_window);

    assert_eq!(context_window, 4096, "El context window final debe ser 4096");
    println!("✅ Reto 2 superado!\n");

    // -------------------------------------------------------------------------
    // RETO 3: Partición de Dataset (Train, Val, Test) con Constantes
    // -------------------------------------------------------------------------
    println!("--- Reto 3: Partición de Dataset con Constantes ---");
    // Contexto: Tienes un dataset con 50,000 imágenes. Queremos una división:
    //   - 80% Entrenamiento (Train)
    //   - 10% Validación (Val)
    //   - 10% Prueba (Test)

    const TOTAL_MUESTRAS: usize = 50_000;
    const PORCENTAJE_TRAIN: f32 = 0.80;
    const PORCENTAJE_VAL: f32 = 0.10;

    let muestras_train = (TOTAL_MUESTRAS as f32 * PORCENTAJE_TRAIN) as usize;
    let muestras_val = (TOTAL_MUESTRAS as f32 * PORCENTAJE_VAL) as usize;
    let muestras_test = TOTAL_MUESTRAS - (muestras_train + muestras_val);

    println!("Dataset total : {} muestras", TOTAL_MUESTRAS);
    println!("  - Train (80%): {} muestras", muestras_train);
    println!("  - Val   (10%): {} muestras", muestras_val);
    println!("  - Test  (10%): {} muestras", muestras_test);

    assert_eq!(muestras_train + muestras_val + muestras_test, TOTAL_MUESTRAS);
    println!("✅ Reto 3 superado!\n");

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 01 FUERON COMPLETADOS!");
    println!("========================================================");
}
