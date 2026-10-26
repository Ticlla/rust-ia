//! ==============================================================================
//! CLASE 02: Inferencia de Modelos Exportados con ONNX Runtime (`ort`)
//! ==============================================================================
//!
//! En Inteligencia Artificial moderna, la arquitectura de producción típica es:
//! 1. **Entrenamiento en Python (PyTorch/GPU):** Máxima flexibilidad de investigación.
//! 2. **Exportación a ONNX (Open Neural Network Exchange):** Formato estándar de grafo.
//! 3. **Servicio en Rust con ONNX Runtime (`ort`):** Latencia predecible, cero GIL,
//!    consumo mínimo de memoria y alto rendimiento con SIMD/CUDA.
//!
//! En esta clase aprenderemos:
//! 1. Cómo cargar un grafo `.onnx` en memoria con `ort`.
//! 2. Cómo inspeccionar dinámicamente las entradas y salidas de la red.
//! 3. Cómo pasar tensores de Rust hacia el motor de ONNX sin copias lentas.
//! 4. Cómo funciona el **Dynamic Batching** (lotes de tamaño variable en tiempo de ejecución).

use ort::session::Session;
use ort::value::Tensor;
use std::time::Instant;

// Incrustamos el modelo binario directamente en el ejecutable:
// Modelo: Y = (X * W) + B
// Donde W = [[2.0], [3.0]] y B = [1.5]
// Entradas esperadas: [Batch, 2] floats
// Salidas producidas: [Batch, 1] floats
const MODELO_BYTES: &[u8] = include_bytes!("modelo_linear.onnx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("⚡ CLASE 02: INFERENCIA ONNX RUNTIME EN RUST (ort)");
    println!("============================================================\n");

    println!("--- 1. Inicialización del Entorno ONNX Runtime ---");
    let env = ort::init()
        .with_name("motor_inferencia_rust")
        .build()?;
    println!("✓ Runtime ONNX inicializado correctamente.");

    println!("\n--- 2. Creación de la Sesión de Inferencia ---");
    let mut session = Session::builder(&env)?
        .commit_from_memory(MODELO_BYTES)?;
    println!("✓ Grafo de red neuronal cargado en memoria (186 bytes).");

    // Inspección de entradas y salidas registradas en el modelo:
    println!("\n🔍 Inspección de Arquitectura del Grafo:");
    for (i, input) in session.inputs().iter().enumerate() {
        println!("   • Entrada #{}: Nombre=\"{}\", Tipo={:?}", i, input.name(), input.dtype());
    }
    for (i, output) in session.outputs().iter().enumerate() {
        println!("   • Salida  #{}: Nombre=\"{}\", Tipo={:?}", i, output.name(), output.dtype());
    }

    println!("\n--- 3. Inferencia Individual (Batch Size = 1) ---");
    // Entrada: 1 muestra con 2 variables: [X0 = 10.0, X1 = 5.0]
    // Cálculo esperado: (10.0 * 2.0) + (5.0 * 3.0) + 1.5 = 20.0 + 15.0 + 1.5 = 36.5
    let input_single = Tensor::from_array(([1, 2], vec![10.0f32, 5.0]))?;

    let t0 = Instant::now();
    let resultado_pred = {
        let outputs = session.run(ort::inputs!["input" => input_single])?;
        let (_, resultado_slice) = outputs["output"].try_extract_tensor::<f32>()?;
        resultado_slice[0]
    };
    let duracion_single = t0.elapsed();

    println!("✓ Inferencia completada en: {:.3?}", duracion_single);
    println!("   Entrada: [10.0, 5.0]");
    println!("   Salida del modelo ONNX: {:.2} (Esperado: 36.50)", resultado_pred);

    println!("\n--- 4. Dynamic Batching (Batch Size = 4) ---");
    // Una gran ventaja de ONNX en Rust es que la primera dimensión (Batch) es dinámica ([-1, 2]).
    // Podemos enviar múltiples peticiones en un solo viaje de memoria:
    let batch_datos = vec![
        1.0f32, 2.0, // Muestra 0 -> (1*2 + 2*3) + 1.5 = 9.5
        3.0,    4.0, // Muestra 1 -> (3*2 + 4*3) + 1.5 = 19.5
        0.0,    0.0, // Muestra 2 -> (0*2 + 0*3) + 1.5 = 1.5
        5.0,    1.0, // Muestra 3 -> (5*2 + 1*3) + 1.5 = 14.5
    ];
    let input_batch = Tensor::from_array(([4, 2], batch_datos))?;

    let t1 = Instant::now();
    let res_batch_vec = {
        let outputs_batch = session.run(ort::inputs!["input" => input_batch])?;
        let (_, res_batch_slice) = outputs_batch["output"].try_extract_tensor::<f32>()?;
        res_batch_slice.to_vec()
    };
    let duracion_batch = t1.elapsed();

    println!("✓ Lote dinámico de 4 muestras procesado en: {:.3?}", duracion_batch);
    for i in 0..4 {
        println!("   • Muestra {}: Predicción = {:.2}", i, res_batch_vec[i]);
    }

    println!("\n✅ Demostración teórica de ONNX Runtime completada con éxito.");
    Ok(())
}
