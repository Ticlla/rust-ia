//! ==============================================================================
//! FASE 4: IA & MACHINE LEARNING EN RUST
//! CLASE 06: Checkpoint Final — Servidor de Inferencia LLM con Token Streaming
//! ARCHIVO: 02_errores_comunes.rs
//! ==============================================================================
//!
//! Laboratorio de Errores Comunes en Servidores de Inferencia LLM y Streaming:
//! 1. Stdout Buffering Trap: Olvidar `flush()` dejando la interfaz congelada.
//! 2. Inestabilidad Numérica por Temperatura <= 0 (División por cero -> NaN).
//! 3. Duplicación exponencial en KV-Cache al pasar historial completo en vez de 1 token.
//! 4. Bucle infinito por omisión de condición de parada EOS.
//! ==============================================================================

use candle_core::{Device, Result, Tensor};
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("==============================================================================");
    println!("CLASE 06: LABORATORIO DE ERRORES COMUNES EN SERVING DE LLMs Y STREAMING");
    println!("==============================================================================");

    let dev = Device::Cpu;

    // -------------------------------------------------------------------------
    // ERROR 1: Stdout Buffering Trap (Falta de flush)
    // -------------------------------------------------------------------------
    println!("\n[ERROR 1] El dilema del buffering en streaming:");
    println!("   En Linux/Unix, la salida estándar (stdout) está protegida por un buffer.");
    println!("   Si haces `print!(\"...\")` sin `io::stdout().flush()`, el sistema operativo");
    println!("   retiene los bytes en RAM hasta llenar 4KB o encontrar un '\\n'.");
    println!("   -> CONSECUENCIA: El usuario ve la pantalla congelada y luego todo de golpe.");
    println!("   -> SOLUCIÓN: Tras emitir cada chunk en Rust/Axum/SSE, SIEMPRE ejecutar:");
    println!("      io::stdout().flush().unwrap();");

    // Demostración de flush inmediato
    print!("   -> Streaming verificado: [Generando");
    io::stdout().flush().unwrap();
    print!(" tokens");
    io::stdout().flush().unwrap();
    print!(" en vivo...]\n");
    io::stdout().flush().unwrap();

    // -------------------------------------------------------------------------
    // ERROR 2: Inestabilidad Numérica con Temperatura = 0.0
    // -------------------------------------------------------------------------
    println!("\n[ERROR 2] División por cero al usar Temperatura = 0.0:");
    let logits = Tensor::from_vec(vec![1.5f32, 3.2f32, 0.8f32], 3, &dev)?;
    let bad_temp = 0.0f64;

    // Si dividimos directamente por temperatura 0.0:
    let bad_scaled = (&logits / bad_temp)?;
    let bad_probs = candle_nn::ops::softmax(&bad_scaled, 0)?;
    let bad_vals = bad_probs.to_vec1::<f32>()?;
    println!("   Logits divididos por T=0.0: {:?}", bad_scaled.to_vec1::<f32>()?);
    println!("   Probabilidades resultantes: {:?}", bad_vals);
    if bad_vals.iter().any(|v| v.is_nan()) {
        println!("   -> ALERTA: Salida contiene NaN! El sampler colapsa o entra en pánico.");
        println!("   -> SOLUCIÓN: Si T <= 1e-4, bifurcar directamente a Greedy (Argmax) sin dividir.");
    }

    // -------------------------------------------------------------------------
    // ERROR 3: Duplicación de Contexto en KV-Cache durante Decode
    // -------------------------------------------------------------------------
    println!("\n[ERROR 3] Pasar el prompt completo en el paso de Decode:");
    let past_k = Tensor::zeros((1, 1, 4, 8), candle_core::DType::F32, &dev)?; // 4 tokens
    println!("   Longitud en KV-Cache inicial: 4");

    // ERROR: En vez de enviar solo el nuevo token [1, 1, 1, 8],
    // el desarrollador vuelve a enviar los 5 tokens acumulados [1, 1, 5, 8]:
    let accidental_history_k = Tensor::zeros((1, 1, 5, 8), candle_core::DType::F32, &dev)?;
    let corrupted_k = Tensor::cat(&[&past_k, &accidental_history_k], 2)?;
    println!("   Longitud tras append erróneo: {} (Esperado: 5)", corrupted_k.dim(2)?);
    println!("   -> ALERTA: Duplicación exponencial de memoria! Causa OOM inmediato en GPU.");
    println!("   -> REGLA: En Decode, el tensor de entrada DEBE tener seq_len = 1.");

    // -------------------------------------------------------------------------
    // ERROR 4: Omitir la detección del token EOS (<endoftext> / </s>)
    // -------------------------------------------------------------------------
    println!("\n[ERROR 4] Bucle infinito por omisión de EOS:");
    let eos_id = 2u32;
    let simulated_predictions = [4u32, 6, 2, 8, 9]; // El tercer token es EOS
    let mut generated = Vec::new();

    for &tok in &simulated_predictions {
        if tok == eos_id {
            println!("   -> EOS detectado (Token ID: {}). Deteniendo generación limpiamente.", tok);
            break;
        }
        generated.push(tok);
    }
    println!("   Tokens guardados: {:?}", generated);
    println!("   -> REGLA: Romper el ciclo de decode inmediatamente si next_token == eos_token_id.");

    println!("\n==============================================================================");
    println!("TODOS LOS ERRORES CRÍTICOS DE SERVING HAN SIDO DEMOSTRADOS Y MITIGADOS");
    println!("==============================================================================");
    Ok(())
}
