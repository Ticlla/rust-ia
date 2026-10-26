//! ==============================================================================
//! CLASE 02: Errores Comunes con ONNX Runtime (`ort`)
//! ==============================================================================
//!
//! Al servir modelos de Machine Learning exportados desde Python, los errores
//! más habituales ocurren en los límites entre el grafo ONNX y los tensores de Rust.

use ort::session::Session;
use ort::value::Tensor;

const MODELO_BYTES: &[u8] = include_bytes!("modelo_linear.onnx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones en Inferencia ONNX");
    println!("============================================================\n");

    let env = ort::init().with_name("lab_errores_onnx").build()?;
    let mut session = Session::builder(&env)?
        .commit_from_memory(MODELO_BYTES)?;

    error_1_nombre_de_entrada_incorrecto(&mut session);
    error_2_dimension_de_features_incompatible(&mut session);
    error_3_mantener_el_borrow_de_session_outputs();

    println!("\n✅ Todos los antipatrones analizados y prevenidos con éxito.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Nombre de Entrada Inexistente en el Grafo
/// ----------------------------------------------------------------------------
/// Si el modelo fue exportado con el nombre "input", pero en Rust intentas
/// alimentarlo con "features" o "datos", ONNX Runtime rechazará la petición.
fn error_1_nombre_de_entrada_incorrecto(session: &mut Session) {
    println!("--- Error 1: Nombre de Entrada No Coincidente ---");

    let tensor = Tensor::from_array(([1, 2], vec![1.0f32, 2.0])).unwrap();

    // ❌ Intentamos pasar "datos_usuario" en lugar de "input":
    let resultado = session.run(ort::inputs!["datos_usuario" => tensor]);

    match resultado {
        Ok(_) => println!("❌ No debería haber funcionado."),
        Err(e) => {
            println!("✓ Error capturado limpiamente sin crashear:");
            println!("   {:?}", e);
            println!("💡 Buena práctica: Inspecciona `session.inputs()` al inicio para validar los nombres.\n");
        }
    }
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Dimensiones de Características Incompatibles (Shape Mismatch)
/// ----------------------------------------------------------------------------
/// El modelo espera [Batch, 2] porque la matriz de pesos W tiene forma [2, 1].
/// Si le enviamos [Batch, 3], la multiplicación matricial es físicamente imposible.
fn error_2_dimension_de_features_incompatible(session: &mut Session) {
    println!("--- Error 2: Desalineación de Dimensiones de Entrada (Shape Mismatch) ---");

    // ❌ Enviamos 3 features en vez de 2:
    let tensor_invalido = Tensor::from_array(([1, 3], vec![1.0f32, 2.0, 3.0])).unwrap();
    let resultado = session.run(ort::inputs!["input" => tensor_invalido]);

    match resultado {
        Ok(_) => println!("❌ No debería haber funcionado."),
        Err(e) => {
            println!("✓ ONNX Runtime rechaza la inferencia inválida con Result::Err:");
            println!("   {:?}", e);
            println!("💡 Regla: En Dynamic Batching, el batch puede variar, pero las features deben coincidir.\n");
        }
    }
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Retener el Borrow de `SessionOutputs` (Borrow Checker)
/// ----------------------------------------------------------------------------
/// `session.run(...)` toma un préstamo mutable (`&mut self`) de la sesión.
/// Si guardas la variable `outputs` viva, no podrás ejecutar una segunda inferencia.
fn error_3_mantener_el_borrow_de_session_outputs() {
    println!("--- Error 3: El Borrow de SessionOutputs en Rust ---");
    println!("❌ Antipatrón: Mantener `let outputs = session.run(...)?` vivo mientras llamas a `session.run` otra vez.");
    println!("  Patrón Correcto: Extraer los datos a un `Vec<f32>` o aislar la inferencia en un bloque `{{ ... }}`.");
    println!("   Esto libera el borrow mutable de la sesión inmediatamente para el siguiente request.");
}
