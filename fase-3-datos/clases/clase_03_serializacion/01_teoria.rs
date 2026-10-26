use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use safetensors::serialize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct ModeloConfig {
    pub nombre: String,
    pub dimension_oculta: usize,
    pub tasa_aprendizaje: f32,
    pub pesos: Vec<f32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== CLASE 03: Serialización Binaria de Modelos ===");

    // 1. Demostración Bincode
    let config = ModeloConfig {
        nombre: "MiModelo-v1".to_string(),
        dimension_oculta: 4,
        tasa_aprendizaje: 0.001,
        pesos: vec![0.5, 1.2, -0.8, 2.0],
    };

    let bytes_bincode = bincode::serialize(&config)?;
    let json_str = serde_json::to_string(&config)?;

    println!("Tamaño en JSON:    {} bytes", json_str.len());
    println!("Tamaño en Bincode: {} bytes", bytes_bincode.len());

    let config_deserializada: ModeloConfig = bincode::deserialize(&bytes_bincode)?;
    assert_eq!(config, config_deserializada);
    println!("✓ Bincode serializó y deserializó con exactitud bit a bit.");

    // 2. Demostración SafeTensors (Hugging Face)
    let pesos_raw: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let bytes_f32: &[u8] = bytemuck_or_cast(&pesos_raw);
    let shape = vec![2, 3];
    let tensor_view = TensorView::new(Dtype::F32, shape.clone(), bytes_f32)?;

    let mut tensors = HashMap::new();
    tensors.insert("capa1.pesos", tensor_view);

    let bytes_safetensors = serialize(&tensors, None)?;
    println!("Tamaño SafeTensors: {} bytes", bytes_safetensors.len());

    let safetensors = SafeTensors::deserialize(&bytes_safetensors)?;
    let tensor_leido = safetensors.tensor("capa1.pesos")?;
    println!("Tensor leído de SafeTensors:");
    println!("  Dtype: {:?}", tensor_leido.dtype());
    println!("  Shape: {:?}", tensor_leido.shape());
    println!("  Bytes totales: {}", tensor_leido.data().len());

    Ok(())
}

fn bytemuck_or_cast(slice: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * std::mem::size_of::<f32>(),
        )
    }
}
