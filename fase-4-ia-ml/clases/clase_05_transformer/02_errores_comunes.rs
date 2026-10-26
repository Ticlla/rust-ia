//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 05: Arquitectura Transformer y Memoria KV-Cache
//! ARCHIVO: 02_errores_comunes.rs
//! ==============================================================================
//!
//! Laboratorio de Errores Comunes al implementar Transformers y KV-Cache en Rust:
//! 1. Shape mismatch en Q @ K^T por olvidar transponer o transponer el eje incorrecto.
//! 2. Concatenar el KV-Cache en la dimensión de Heads o Batch en vez de Secuencia.
//! 3. Fuga de causalidad: Usar 0.0 en lugar de -inf en la máscara de atención.
//! 4. Desfase temporal en RoPE: Aplicar rotación con offset = 0 durante el paso de decode.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 05: LABORATORIO DE ERRORES COMUNES EN TRANSFORMERS Y KV-CACHE");
    println!("==============================================================================");

    let dev = Device::Cpu;

    // -------------------------------------------------------------------------
    // ERROR 1: Olvidar transponer K antes del producto punto de atención
    // -------------------------------------------------------------------------
    println!("\n[ERROR 1] MatMul directo de Q @ K sin transponer K:");
    let q = Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &dev)?;
    let k = Tensor::zeros((1, 2, 4, 8), candle_core::DType::F32, &dev)?;

    // Q tiene shape [1, 2, 4, 8]. K tiene shape [1, 2, 4, 8].
    // Para multiplicar matrices internas: (4 x 8) @ (4 x 8) es matemáticamente INVÁLIDO.
    // Requiere (4 x 8) @ (8 x 4) -> transponer dim 2 y 3.
    match q.matmul(&k) {
        Ok(_) => println!("   Inesperado: Multiplicación permitida"),
        Err(e) => {
            println!("   -> ERROR CAPTURADO (esperado): {}", e);
            println!("   -> SOLUCIÓN: Usar `k.transpose(2, 3)?` para obtener shape [1, 2, 8, 4].");
            let k_t = k.transpose(2, 3)?;
            let correct = q.matmul(&k_t)?;
            println!("   -> Resultado corregido shape: {:?}", correct.shape());
        }
    }

    // -------------------------------------------------------------------------
    // ERROR 2: Concatenar el KV-Cache en la dimensión equivocada (Heads en vez de Seq)
    // -------------------------------------------------------------------------
    println!("\n[ERROR 2] Concatenar nuevo token en dim = 1 (Heads) en vez de dim = 2 (Seq):");
    let past_k = Tensor::zeros((1, 4, 10, 16), candle_core::DType::F32, &dev)?;
    let new_k = Tensor::zeros((1, 4, 1, 16), candle_core::DType::F32, &dev)?;

    // Si concatenamos en dim 1 (cabezas):
    // past_k: [1, 4, 10, 16], new_k: [1, 4, 1, 16] -> dim 2 no coincide (10 != 1)!
    match Tensor::cat(&[&past_k, &new_k], 1) {
        Ok(_) => println!("   Inesperado: Concatenación exitosa"),
        Err(e) => {
            println!("   -> ERROR CAPTURADO (esperado): {}", e);
            println!("   -> CAUSA: Intentar concatenar cabezas cuando las longitudes de secuencia difieren.");
            println!("   -> SOLUCIÓN: El KV-Cache SIEMPRE se acumula a lo largo de la secuencia (dim 2).");
            let correct_cat = Tensor::cat(&[&past_k, &new_k], 2)?;
            println!("   -> Shape corregido tras append: {:?}", correct_cat.shape());
        }
    }

    // -------------------------------------------------------------------------
    // ERROR 3: Usar 0.0 en lugar de -inf en la Máscara Causal
    // -------------------------------------------------------------------------
    println!("\n[ERROR 3] Fuga de información temporal: Enmascarar con 0.0 en vez de -inf:");
    // Supongamos un score de atención no normalizado: [2.0, 5.0] donde el segundo token es futuro.
    // Si queremos bloquear el segundo token:
    let scores = Tensor::from_vec(vec![2.0f32, 5.0f32], (1, 2), &dev)?;

    // Enmascaramiento INCORRECTO: sumar 0.0 al futuro
    let bad_mask = Tensor::from_vec(vec![0.0f32, 0.0f32], (1, 2), &dev)?;
    let bad_scores = scores.broadcast_add(&bad_mask)?;
    let bad_probs = candle_nn::ops::softmax(&bad_scores, 1)?;
    println!("   Scores con máscara incorrecta (0.0): {:?}", bad_probs.to_vec2::<f32>()?);
    println!("   -> ALERTA: El token futuro recibe {:.2}% de probabilidad! No fue bloqueado.", 
        bad_probs.to_vec2::<f32>()?[0][1] * 100.0);

    // Enmascaramiento CORRECTO: sumar -inf al futuro
    let good_mask = Tensor::from_vec(vec![0.0f32, f32::NEG_INFINITY], (1, 2), &dev)?;
    let good_scores = scores.broadcast_add(&good_mask)?;
    let good_probs = candle_nn::ops::softmax(&good_scores, 1)?;
    println!("   Scores con máscara correcta (-inf):  {:?}", good_probs.to_vec2::<f32>()?);
    println!("   -> ÉXITO: exp(-inf) = 0.0. El token futuro recibe exactamente 0.0% de atención.");

    // -------------------------------------------------------------------------
    // ERROR 4: Desfase posicional en RoPE al generar con KV-Cache
    // -------------------------------------------------------------------------
    println!("\n[ERROR 4] Desfase posicional en RoPE durante Decode:");
    println!("   Al procesar el 5to token (índice 4), si se pasa offset = 0:");
    println!("   El modelo asumirá que el nuevo token es la primera palabra del prompt!");
    println!("   Consecuencia: Pérdida total de coherencia sintáctica y bucles infinitos en el LLM.");
    println!("   -> REGLA DE ORO: En cada llamada a decode, offset = cache.seq_len().");

    println!("\n==============================================================================");
    println!("TODOS LOS CASOS DE ERROR HAN SIDO DIAGNOSTICADOS Y RESUELTOS");
    println!("==============================================================================");
    Ok(())
}
