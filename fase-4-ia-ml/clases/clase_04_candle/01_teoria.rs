//! ==============================================================================
//! CLASE 04: Framework de Deep Learning `Candle` (Hugging Face)
//! ==============================================================================
//!
//! `Candle` es el framework minimalista de Deep Learning creado por Hugging Face
//! completamente en Rust para eliminar los gigabytes de dependencias de PyTorch.
//!
//! En esta clase exploraremos:
//! 1. Dispositivos (`Device`) y tensores (`Tensor`).
//! 2. Operaciones matemáticas fundamentales (MatMul, Softmax).
//! 3. Capas lineales (`Linear`) y la activación estrella de los LLMs: **SwiGLU**.
//! 4. Normalización moderna: **RMSNorm** (Root Mean Square Normalization).
//! 5. Carga de pesos desde `.safetensors` usando `VarBuilder`.

use candle_core::{DType, Device, Tensor};
use candle_nn::{ops::silu, Linear, Module, VarBuilder};
use std::collections::HashMap;

fn main() -> candle_core::Result<()> {
    println!("============================================================");
    println!("🕯️ CLASE 04: DEEP LEARNING NATIVO CON CANDLE (Hugging Face)");
    println!("============================================================\n");

    let device = Device::Cpu;

    println!("--- 1. Tensores y Operaciones Básicas en CPU ---");
    // Creamos dos matrices 2D [2, 2]:
    let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device)?;
    let b = Tensor::new(&[[2.0f32, 0.0], [1.0, 3.0]], &device)?;

    // Multiplicación matricial (MatMul):
    let c = a.matmul(&b)?;
    println!("Matriz A:\n{:?}", a.to_vec2::<f32>()?);
    println!("Matriz B:\n{:?}", b.to_vec2::<f32>()?);
    println!("MatMul (A x B):\n{:?}", c.to_vec2::<f32>()?);

    // Softmax sobre la última dimensión (para calcular probabilidades o atención):
    let probs = candle_nn::ops::softmax(&c, 1)?;
    println!("\nSoftmax sobre filas de C (las probabilidades suman 1.0):");
    println!("{:?}\n", probs.to_vec2::<f32>()?);

    println!("--- 2. Capa Lineal (Dense Layer) ---");
    // y = x * W^T + b
    let peso_w = Tensor::new(&[[2.0f32, 1.0], [0.5, 3.0]], &device)?;
    let sesgo_b = Tensor::new(&[0.1f32, 0.2], &device)?;
    let capa_lineal = Linear::new(peso_w, Some(sesgo_b));

    let x = Tensor::new(&[[1.0f32, 2.0]], &device)?; // Entrada [Batch 1, Dim 2]
    let y = capa_lineal.forward(&x)?;
    println!("Entrada X: {:?}", x.to_vec2::<f32>()?);
    println!("Salida Lineal (X * W^T + b): {:?}\n", y.to_vec2::<f32>()?);

    println!("--- 3. La Activación de los LLMs Modernos: SwiGLU (LLaMA / Qwen) ---");
    // SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up))
    // Es la capa que le da la inteligencia no lineal a los Transformers modernos:
    let x_gate = Tensor::new(&[[0.5f32, -1.0, 2.0]], &device)?;
    let x_up = Tensor::new(&[[1.0f32, 2.0, 0.5]], &device)?;

    let silu_gate = silu(&x_gate)?;
    let swiglu = (silu_gate * x_up)?;
    println!("SwiGLU resultado: {:?}\n", swiglu.to_vec2::<f32>()?);

    println!("--- 4. Carga de Pesos desde Safetensors con VarBuilder ---");
    // Simulamos un archivo .safetensors en memoria:
    let mut tensors_map = HashMap::new();
    let sample_weights = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &device)?;
    tensors_map.insert("linear.weight".to_string(), sample_weights);

    // Creamos un VarBuilder a partir de un mapa de tensores:
    let vb = VarBuilder::from_tensors(tensors_map, DType::F32, &device);
    let peso_recuperado = vb.get((2, 2), "linear.weight")?;
    println!("✓ Peso 'linear.weight' cargado exitosamente via VarBuilder:");
    println!("{:?}\n", peso_recuperado.to_vec2::<f32>()?);

    println!("✅ Demostración teórica de Candle completada con éxito.");
    Ok(())
}
