//! ==============================================================================
//! CLASE 03: Ejercicios Prácticos de Serialización para Modelos de IA
//! ==============================================================================
//!
//! En este laboratorio resolverás 3 ejercicios de almacenamiento de modelos:
//! 1. Guardar y Restaurar Checkpoints completos con Bincode.
//! 2. Serializar pesos de capas en formato estándar SafeTensors.
//! 3. Leer tensores de SafeTensors y reconstruir matrices `Array2<f32>` de ndarray.

use ndarray::Array2;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use safetensors::serialize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 03: Serialización de Modelos y Safetensors");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c03_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Checkpoints de Entrenamiento con Bincode
/// ----------------------------------------------------------------------------
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct CheckpointEntrenamiento {
    pub epoca: usize,
    pub loss: f32,
    pub lr: f32,
    pub pesos: Vec<f32>,
}

pub fn serializar_checkpoint(cp: &CheckpointEntrenamiento) -> Result<Vec<u8>, bincode::Error> {
    bincode::serialize(cp)
}

pub fn deserializar_checkpoint(bytes: &[u8]) -> Result<CheckpointEntrenamiento, bincode::Error> {
    bincode::deserialize(bytes)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Exportar Tensores a Formato SafeTensors de Hugging Face
/// ----------------------------------------------------------------------------
/// Serializa un tensor 2D de floats con su forma dada en formato .safetensors.
pub fn exportar_tensor_safetensors(
    nombre_tensor: &str,
    datos: &[f32],
    shape: Vec<usize>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let bytes_f32: &[u8] = unsafe {
        std::slice::from_raw_parts(
            datos.as_ptr() as *const u8,
            datos.len() * std::mem::size_of::<f32>(),
        )
    };

    let tensor_view = TensorView::new(Dtype::F32, shape, bytes_f32)?;
    let mut mapa = HashMap::new();
    mapa.insert(nombre_tensor, tensor_view);

    let bytes = serialize(&mapa, None)?;
    Ok(bytes)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Leer SafeTensors y Reconstruir Tensor de ndarray
/// ----------------------------------------------------------------------------
/// Lee un buffer en formato SafeTensors, extrae el tensor indicado y
/// lo devuelve como un `Array2<f32>` de `ndarray`.
pub fn cargar_tensor_a_ndarray(
    bytes_safetensors: &[u8],
    nombre_tensor: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let st = SafeTensors::deserialize(bytes_safetensors)?;
    let tensor_info = st.tensor(nombre_tensor)?;

    let shape = tensor_info.shape();
    if shape.len() != 2 {
        return Err("El tensor no es 2D".into());
    }

    let n_filas = shape[0];
    let n_cols = shape[1];

    let bytes_data = tensor_info.data();
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bytes_data.as_ptr() as *const f32,
            bytes_data.len() / std::mem::size_of::<f32>(),
        )
    };

    let array2 = Array2::from_shape_vec((n_filas, n_cols), floats.to_vec())?;
    Ok(array2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_bincode_roundtrip() {
        let cp = CheckpointEntrenamiento {
            epoca: 42,
            loss: 0.0125,
            lr: 0.0003,
            pesos: vec![1.5, -2.3, 0.45, 9.1],
        };

        let bytes = serializar_checkpoint(&cp).unwrap();
        let restaurado = deserializar_checkpoint(&bytes).unwrap();

        assert_eq!(cp, restaurado);
    }

    #[test]
    fn test_safetensors_exportar_y_cargar() {
        let pesos = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3]; // Matriz 2x3

        let bytes = exportar_tensor_safetensors("encoder.layer1.weight", &pesos, shape).unwrap();
        assert!(!bytes.is_empty());

        let tensor_ndarray = cargar_tensor_a_ndarray(&bytes, "encoder.layer1.weight").unwrap();

        assert_eq!(tensor_ndarray.shape(), &[2, 3]);
        assert_eq!(tensor_ndarray[[0, 0]], 1.0);
        assert_eq!(tensor_ndarray[[0, 2]], 3.0);
        assert_eq!(tensor_ndarray[[1, 0]], 4.0);
        assert_eq!(tensor_ndarray[[1, 2]], 6.0);
    }
}
