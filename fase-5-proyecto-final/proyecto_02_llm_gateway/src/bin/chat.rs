//! Chat Interactivo en Tiempo Real con SmolLM2-135M en Rust

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{Cache, Config as LlamaConfig, Llama};
use std::io::{self, BufRead, Write};
use std::time::Instant;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("==============================================================================");
    println!("💬 CHAT INTERACTIVO CON MODELO REAL (SMOLLM2-135M) EN RUST");
    println!("==============================================================================");
    println!("Cargando modelo en memoria...");

    let dev = Device::Cpu;
    let model_dir = "/home/alcidesticlla/Documents/rust-ia/models/smollm2-135m";
    let weights_path = format!("{}/model.safetensors", model_dir);
    let tokenizer_path = format!("{}/tokenizer.json", model_dir);

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

    let start_load = Instant::now();
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &dev)? };
    let model = Llama::load(vb, &config)?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| e.to_string())?;
    println!(
        "¡Modelo cargado en {:.2} ms!",
        start_load.elapsed().as_secs_f64() * 1000.0
    );

    println!("\nEl modelo está listo para responder tus preguntas.");
    println!("Escribe lo que quieras (o 'salir' / 'exit' para terminar).\n");

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();

    loop {
        print!("Tú > ");
        io::stdout().flush().unwrap();

        let user_text = match lines.next() {
            Some(Ok(text)) => text.trim().to_string(),
            _ => break,
        };

        if user_text.is_empty() {
            continue;
        }

        if user_text.eq_ignore_ascii_case("salir")
            || user_text.eq_ignore_ascii_case("exit")
            || user_text.eq_ignore_ascii_case("quit")
        {
            println!("\n¡Hasta pronto!");
            break;
        }

        // Formato ChatML
        let prompt = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_text
        );
        let encoded = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| e.to_string())?;
        let prompt_ids = encoded.get_ids();
        let prompt_len = prompt_ids.len();

        let mut cache = Cache::new(true, DType::F32, &config, &dev)?;

        // 1. Prefill
        let input_tensor = Tensor::new(&prompt_ids[..], &dev)?.unsqueeze(0)?;
        let logits = model.forward(&input_tensor, 0, &mut cache)?;
        let last_logits = logits.squeeze(0)?.squeeze(0)?;
        let mut next_token = sample_token(&last_logits, 0.6)?;

        // 2. Decode con Streaming en Vivo
        print!("SmolLM > ");
        io::stdout().flush().unwrap();

        let max_tokens = 80;
        let eos_id = 2u32;
        let im_end_id = 49153u32;

        for step in 0..max_tokens {
            if next_token == eos_id || next_token == im_end_id {
                break;
            }

            let chunk = tokenizer.decode(&[next_token], false).unwrap_or_default();
            print!("{}", chunk);
            io::stdout().flush().unwrap();

            let input_token = Tensor::new(&[next_token], &dev)?.unsqueeze(0)?;
            let logits = model.forward(&input_token, prompt_len + step, &mut cache)?;
            let token_logits = logits.squeeze(0)?.squeeze(0)?;
            next_token = sample_token(&token_logits, 0.6)?;
        }
        println!("\n");
    }

    Ok(())
}

fn sample_token(logits: &Tensor, temperature: f64) -> candle_core::Result<u32> {
    if temperature <= 0.001 {
        let vals = logits.to_vec1::<f32>()?;
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (i, &v) in vals.iter().enumerate() {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }
        return Ok(max_idx as u32);
    }

    let scaled = (logits / temperature)?;
    let probs = candle_nn::ops::softmax(&scaled, 0)?;
    let vals = probs.to_vec1::<f32>()?;

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
