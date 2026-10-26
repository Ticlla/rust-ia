//! ==============================================================================
//! CLASE 04: Errores Comunes con `Candle` (Deep Learning en Rust)
//! ==============================================================================
//!
//! Al construir redes neuronales con Candle, los errores más frecuentes
//! están relacionados con incompatibilidades de formas, tipos de datos o dispositivos.

use candle_core::{DType, Device, Tensor};

fn main() -> candle_core::Result<()> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones de Deep Learning");
    println!("============================================================\n");

    let device = Device::Cpu;

    error_1_incompatibilidad_de_dtypes(&device)?;
    error_2_desalineacion_en_matmul(&device)?;
    error_3_carga_de_peso_inexistente_en_varbuilder(&device)?;

    println!("\n✅ Todos los antipatrones analizados y prevenidos con éxito.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Operación entre DTypes Incompatibles (F32 vs F64)
/// ----------------------------------------------------------------------------
/// En Rust y Candle, no existe conversión mágica implícita entre F32 y F64.
/// Si intentas sumar o multiplicar tensores de diferente precisión,
/// Candle devuelve un error seguro `DTypeMismatch`.
fn error_1_incompatibilidad_de_dtypes(device: &Device) -> candle_core::Result<()> {
    println!("--- Error 1: Conflicto de Tipos de Dato (DTypeMismatch) ---");

    let tensor_f32 = Tensor::new(&[1.0f32, 2.0], device)?;
    let tensor_u32 = Tensor::new(&[3u32, 4], device)?;

    // ❌ Intentamos sumar F32 con U32 directamente:
    let resultado = tensor_f32.add(&tensor_u32);

    match resultado {
        Ok(_) => println!("❌ No debería haber tenido éxito."),
        Err(e) => {
            println!("✓ Candle previene mezcla silenciosa de tipos de dato:");
            println!("   Error controlado: {:?}\n", e);
        }
    }
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Desalineación de Dimensiones Interiores en MatMul
/// ----------------------------------------------------------------------------
/// Para multiplicar una matriz A [M, K] por una matriz B [P, N],
/// la regla matemática estricta exige que K == P.
fn error_2_desalineacion_en_matmul(device: &Device) -> candle_core::Result<()> {
    println!("--- Error 2: Dimensiones Incompatibles en MatMul ---");

    let a = Tensor::new(&[[1.0f32, 2.0, 3.0]], device)?; // [1, 3]
    let b = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], device)?; // [2, 2]

    // ❌ Multiplicamos [1, 3] x [2, 2]: (3 != 2)
    let resultado = a.matmul(&b);

    match resultado {
        Ok(_) => println!("❌ No debería haber tenido éxito."),
        Err(e) => {
            println!("✓ Error de dimensiones capturado en tiempo de ejecución:");
            println!("   Error: {:?}\n", e);
        }
    }
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Peso Faltante o Mal Nombrado en VarBuilder
/// ----------------------------------------------------------------------------
/// Si tu modelo busca "transformer.layers.0.weight" pero en el archivo .safetensors
/// se guardó como "layers.0.weight", VarBuilder devuelve un error `CannotFindTensor`.
fn error_3_carga_de_peso_inexistente_en_varbuilder(device: &Device) -> candle_core::Result<()> {
    println!("--- Error 3: Tensor Inexistente en VarBuilder ---");

    let tensors_map = std::collections::HashMap::new();
    let vb = candle_nn::VarBuilder::from_tensors(tensors_map, DType::F32, device);

    // ❌ Buscamos una capa que no existe en el mapa:
    let resultado = vb.get((2, 2), "capa_inexistente.weight");

    match resultado {
        Ok(_) => println!("❌ No debería haber tenido éxito."),
        Err(e) => {
            println!("✓ VarBuilder detecta la ausencia del tensor en safetensors:");
            println!("   Error: {:?}\n", e);
        }
    }
    Ok(())
}
