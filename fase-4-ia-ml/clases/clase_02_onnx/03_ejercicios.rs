//! ==============================================================================
//! CLASE 02: Ejercicios Prácticos de Inferencia con ONNX Runtime (`ort`)
//! ==============================================================================
//!
//! En este laboratorio implementarás y validarás:
//! 1. `ejecutar_inferencia_individual`: Cargar modelo y procesar una muestra.
//! 2. `ejecutar_dynamic_batching`: Procesar un lote de N muestras en una sola llamada.
//! 3. `inspeccionar_entradas`: Extraer metadatos de nombres y formas del grafo.

use ort::session::Session;
use ort::value::Tensor;

pub const MODELO_BYTES: &[u8] = include_bytes!("modelo_linear.onnx");

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 02: Inferencia con ONNX Runtime");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c02_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Inferencia Individual
/// ----------------------------------------------------------------------------
/// Dado un vector de 2 características [x0, x1], ejecuta la inferencia
/// y devuelve la predicción escalar única producida por el modelo.
pub fn inferencia_individual(
    session: &mut Session,
    x0: f32,
    x1: f32,
) -> Result<f32, Box<dyn std::error::Error>> {
    let tensor = Tensor::from_array(([1, 2], vec![x0, x1]))?;
    let outputs = session.run(ort::inputs!["input" => tensor])?;
    let (_, slice) = outputs["output"].try_extract_tensor::<f32>()?;
    Ok(slice[0])
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Inferencia por Lotes Dinámicos (Dynamic Batching)
/// ----------------------------------------------------------------------------
/// Dado un vector plano de datos con `batch_size` muestras y 2 características
/// por muestra, ejecuta una sola pasada en el motor y retorna el vector de resultados.
pub fn inferencia_lote(
    session: &mut Session,
    batch_size: usize,
    datos_planos: Vec<f32>,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    assert_eq!(datos_planos.len(), batch_size * 2, "Datos no coinciden con batch_size * 2");

    let tensor = Tensor::from_array(([batch_size, 2], datos_planos))?;
    let outputs = session.run(ort::inputs!["input" => tensor])?;
    let (_, slice) = outputs["output"].try_extract_tensor::<f32>()?;
    Ok(slice.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crear_sesion_prueba() -> Session {
        let env = ort::init().with_name("test_env").build().unwrap();
        Session::builder(&env)
            .unwrap()
            .commit_from_memory(MODELO_BYTES)
            .unwrap()
    }

    #[test]
    fn test_inferencia_individual() -> Result<(), Box<dyn std::error::Error>> {
        let mut session = crear_sesion_prueba();

        // Para [10.0, 5.0]: (10*2 + 5*3) + 1.5 = 20 + 15 + 1.5 = 36.5
        let pred = inferencia_individual(&mut session, 10.0, 5.0)?;
        assert!((pred - 36.5).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_dynamic_batching() -> Result<(), Box<dyn std::error::Error>> {
        let mut session = crear_sesion_prueba();

        // Lote de 3 muestras:
        // Muestra 0: [1.0, 1.0] -> 1*2 + 1*3 + 1.5 = 6.5
        // Muestra 1: [2.0, 2.0] -> 2*2 + 2*3 + 1.5 = 11.5
        // Muestra 2: [3.0, 3.0] -> 3*2 + 3*3 + 1.5 = 16.5
        let datos = vec![1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0];
        let preds = inferencia_lote(&mut session, 3, datos)?;

        assert_eq!(preds.len(), 3);
        assert!((preds[0] - 6.5).abs() < 1e-4);
        assert!((preds[1] - 11.5).abs() < 1e-4);
        assert!((preds[2] - 16.5).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_inspeccion_metadata_grafo() {
        let session = crear_sesion_prueba();
        let inputs = session.inputs();
        let outputs = session.outputs();

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name(), "input");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name(), "output");
    }
}
