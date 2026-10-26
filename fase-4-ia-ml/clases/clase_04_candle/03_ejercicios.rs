//! ==============================================================================
//! CLASE 04: Ejercicios Prácticos de Deep Learning con `Candle`
//! ==============================================================================
//!
//! En este laboratorio implementarás y validarás los componentes de un LLM:
//! 1. `ejecutar_capa_lineal`: Forward pass de una capa densa $y = x W^T + b$.
//! 2. `calcular_swiglu`: La activación de LLaMA y Qwen.
//! 3. `calcular_rms_norm`: La normalización moderna de Transformers.

use candle_core::Tensor;
use candle_nn::{ops::silu, Linear, Module};

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Capa Lineal (Dense Layer)
/// ----------------------------------------------------------------------------
pub fn capa_lineal_forward(
    x: &Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
) -> candle_core::Result<Tensor> {
    let capa = Linear::new(weight, bias);
    capa.forward(x)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Activación SwiGLU (Swish Gated Linear Unit)
/// ----------------------------------------------------------------------------
/// SwiGLU(x_gate, x_up) = SiLU(x_gate) * x_up
pub fn calcular_swiglu(x_gate: &Tensor, x_up: &Tensor) -> candle_core::Result<Tensor> {
    let gate_activado = silu(x_gate)?;
    gate_activado * x_up
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Normalización RMSNorm (Root Mean Square Normalization)
/// ----------------------------------------------------------------------------
/// RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps)
pub fn calcular_rms_norm(x: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    // 1. Elevar al cuadrado x^2
    let x_sq = x.sqr()?;

    // 2. Calcular la media sobre la última dimensión: mean(x^2)
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1)?;

    // 3. Sumar epsilon y sacar raíz cuadrada: sqrt(mean + eps)
    let norm = (mean_sq + eps)?.sqrt()?;

    // 4. Dividir x entre la norma
    x.broadcast_div(&norm)
}

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 04: Deep Learning con Candle");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c04_ejercicios' para evaluar tus soluciones.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_capa_lineal_forward() -> candle_core::Result<()> {
        let device = Device::Cpu;

        // X: [1, 2] con valores [1.0, 2.0]
        // W: [2, 2] con valores [[2.0, 1.0], [0.5, 3.0]]
        // B: [2] con valores [0.1, 0.2]
        let x = Tensor::new(&[[1.0f32, 2.0]], &device)?;
        let w = Tensor::new(&[[2.0f32, 1.0], [0.5, 3.0]], &device)?;
        let b = Some(Tensor::new(&[0.1f32, 0.2], &device)?);

        let out = capa_lineal_forward(&x, w, b)?;

        // y[0] = (1.0*2.0 + 2.0*1.0) + 0.1 = 4.1
        // y[1] = (1.0*0.5 + 2.0*3.0) + 0.2 = 6.7
        let vals = out.to_vec2::<f32>()?;
        assert!((vals[0][0] - 4.1).abs() < 1e-4);
        assert!((vals[0][1] - 6.7).abs() < 1e-4);
        Ok(())
    }

    #[test]
    fn test_swiglu_calculo() -> candle_core::Result<()> {
        let device = Device::Cpu;

        // Para x_gate = 0.0: silu(0.0) = 0.0 * sigmoid(0.0) = 0.0
        // Para x_up = 5.0: swiglu = 0.0 * 5.0 = 0.0
        let x_gate = Tensor::new(&[[0.0f32]], &device)?;
        let x_up = Tensor::new(&[[5.0f32]], &device)?;

        let out = calcular_swiglu(&x_gate, &x_up)?;
        let vals = out.to_vec2::<f32>()?;

        assert!((vals[0][0] - 0.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_rms_norm() -> candle_core::Result<()> {
        let device = Device::Cpu;

        // Vector [3.0, 4.0]:
        // mean(3^2 + 4^2) = mean(9 + 16) = 25 / 2 = 12.5
        // sqrt(12.5) = 3.5355339
        // norm[0] = 3.0 / 3.5355 = 0.8485
        // norm[1] = 4.0 / 3.5355 = 1.1313
        let x = Tensor::new(&[[3.0f32, 4.0]], &device)?;
        let out = calcular_rms_norm(&x, 1e-5)?;

        let vals = out.to_vec2::<f32>()?;
        assert!((vals[0][0] - (3.0 / 12.5f32.sqrt())).abs() < 1e-4);
        assert!((vals[0][1] - (4.0 / 12.5f32.sqrt())).abs() < 1e-4);
        Ok(())
    }
}
