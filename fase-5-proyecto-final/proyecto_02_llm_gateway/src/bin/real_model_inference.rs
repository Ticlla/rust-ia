//! Inferencia Real con Pesos de SmolLM2-135M en Rust usando Candle

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use std::io::{self, Write};
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==============================================================================");
    println!("🚀 INFERENCIA CON MODELO REAL SMOLLM2-135M EN RUST (CANDLE + CPU)");
    println!("==============================================================================");

    let dev = Device::Cpu;
    let model_dir = "/home/alcidesticlla/Documents/rust-ia/models/smollm2-135m";
    let weights_path = format!("{}/model.safetensors", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);

    println!("[1] Configurando hiperparámetros de SmolLM2-135M...");
    let config = LlamaConfig {
        hidden_size: 576,
        intermediate_size: 1536,
        vocab_size: 49152,
        num_hidden_layers: 30,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        use_flash_attn: false,
        rms_norm_eps: 1e-5,
        rope_theta: 100000.0,
        bos_token_id: Some(1),
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: 8192,
        tie_word_embeddings: true,
    };

    println!("[2] Mapeando pesos Safetensors con Zero-Copy (mmap)...");
    let load_start = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &dev)? };
    let model = Llama::load(vb, &config)?;
    println!("    -> Modelo cargado exitosamente en {:.2} ms!", load_start.elapsed().as_secs_f64() * 1000.0);

    println!("[3] Cargando tokenizador Hugging Face...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;

    // Prompt en formato ChatML
    let prompt = "<|im_start|>user\n¿Qué es Rust y por qué es tan rápido?<|im_end|>\n<|im_start|>assistant\n";
    let encoded = tokenizer.encode(prompt, true).map_err(|e| e.to_string())?;
    let prompt_ids = encoded.get_ids();
    let prompt_len = prompt_ids.len();

    println!("\nPrompt: \"{}\"", prompt);
    println!("Tokens del prompt ({} tokens): {:?}", prompt_len, prompt_ids);

    let mut cache = Cache::new(true, DType::F32, &config, &dev)?;

    // -------------------------------------------------------------------------
    // FASE PREFILL
    // -------------------------------------------------------------------------
    println!("\n[4] Ejecutando Fase PREFILL...");
    let prefill_start = Instant::now();
    let input_tensor = Tensor::new(&prompt_ids[..], &dev)?.unsqueeze(0)?;
    let logits = model.forward(&input_tensor, 0, &mut cache)?;
    let ttft = prefill_start.elapsed();

    println!("    -> TTFT (Time To First Token): {:.2} ms", ttft.as_secs_f64() * 1000.0);

    // Muestreo del primer token generado
    let last_logits = logits.squeeze(0)?.squeeze(0)?;
    let mut next_token = sample_greedy(&last_logits)?;

    // -------------------------------------------------------------------------
    // FASE DECODE CON TOKEN STREAMING
    // -------------------------------------------------------------------------
    println!("\n[5] Respuesta generada en tiempo real:\n");
    print!("Asistente: ");
    io::stdout().flush().unwrap();

    let max_new_tokens = 40;
    let eos_token_id = 2u32;
    let im_end_id = 49153u32;
    let mut generated_count = 0;
    let decode_start = Instant::now();

    for step in 0..max_new_tokens {
        if next_token == eos_token_id || next_token == im_end_id {
            break;
        }

        generated_count += 1;
        let token_chunk = tokenizer.decode(&[next_token], false).unwrap_or_default();
        print!("{}", token_chunk);
        io::stdout().flush().unwrap();

        // Decode: enviar solo el nuevo token individual
        let input_token = Tensor::new(&[next_token], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&input_token, prompt_len + step, &mut cache)?;
        let token_logits = logits.squeeze(0)?.squeeze(0)?;
        next_token = sample_greedy(&token_logits)?;
    }
    println!("\n");

    let total_decode_time = decode_start.elapsed();
    let tpot = if generated_count > 0 {
        total_decode_time.as_secs_f64() * 1000.0 / (generated_count as f64)
    } else {
        0.0
    };

    println!("==============================================================================");
    println!("MÉTRICAS REALES EN TU INTEL CORE I5:");
    println!("- Tokens generados: {}", generated_count);
    println!("- TPOT (Time Per Output Token): {:.2} ms/token", tpot);
    println!("- Throughput: {:.2} tokens/segundo", if tpot > 0.0 { 1000.0 / tpot } else { 0.0 });
    println!("==============================================================================");

    Ok(())
}

fn sample_greedy(logits: &Tensor) -> candle_core::Result<u32> {
    let vals = logits.to_vec1::<f32>()?;
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in vals.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    Ok(max_idx as u32)
}
