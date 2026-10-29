//! Herramienta de consola interactiva para probar los modelos de visión y audio.

use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use proyecto_04_edge_inference::audio::{
    AudioPostprocessor, AudioResampler, MelSpectrogramExtractor, SpectrogramConfig, WavDecoder,
};
use proyecto_04_edge_inference::config::EdgeServiceConfig;
use proyecto_04_edge_inference::engine::{ModelKind, SessionPool};
use proyecto_04_edge_inference::vision::{VisionPostprocessor, VisionPreprocessor};

const DEFAULT_VISION_MODEL: &[u8] = include_bytes!("../../models/vision_classifier.onnx");
const DEFAULT_AUDIO_MODEL: &[u8] = include_bytes!("../../models/audio_classifier.onnx");

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = EdgeServiceConfig::default();
    println!("⚡ Inicializando pool de inferencia ONNX...");
    let pool = SessionPool::new(&config, DEFAULT_VISION_MODEL, DEFAULT_AUDIO_MODEL)?;
    let preprocessor = VisionPreprocessor::default();
    let postprocessor = VisionPostprocessor::default();
    let spec_config = SpectrogramConfig::default();
    let extractor = MelSpectrogramExtractor::new(spec_config.clone());
    let audio_postprocessor = AudioPostprocessor::default();

    loop {
        println!();
        println!(
            "================================================================================"
        );
        println!("  🔬 CONSOLA INTERACTIVA EDGE AI — VISIÓN & AUDIO");
        println!(
            "================================================================================"
        );
        println!("  1. [Visión] Analizar imagen desde archivo personalizado (PNG/JPG)");
        println!("  2. [Visión] Probar muestra: 'defecto_inspeccion.png'");
        println!("  3. [Visión] Probar muestra: 'pieza_limpia.png'");
        println!("  4. [Audio]  Analizar audio desde archivo personalizado (.wav)");
        println!("  5. [Audio]  Probar muestra: 'alerta_parada.wav'");
        println!("  6. Salir");
        println!(
            "--------------------------------------------------------------------------------"
        );
        print!("  Selecciona una opción [1-6]: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let choice = input.trim();

        match choice {
            "1" => {
                print!("  Ingresa la ruta de la imagen: ");
                io::stdout().flush()?;
                let mut path_str = String::new();
                io::stdin().read_line(&mut path_str)?;
                let path = path_str.trim();
                run_vision_analysis(path, &preprocessor, &pool, &postprocessor).await;
            }
            "2" => {
                let path = "samples/defecto_inspeccion.png";
                run_vision_analysis(path, &preprocessor, &pool, &postprocessor).await;
            }
            "3" => {
                let path = "samples/pieza_limpia.png";
                run_vision_analysis(path, &preprocessor, &pool, &postprocessor).await;
            }
            "4" => {
                print!("  Ingresa la ruta del archivo WAV: ");
                io::stdout().flush()?;
                let mut path_str = String::new();
                io::stdin().read_line(&mut path_str)?;
                let path = path_str.trim();
                run_audio_analysis(path, &spec_config, &extractor, &pool, &audio_postprocessor)
                    .await;
            }
            "5" => {
                let path = "samples/alerta_parada.wav";
                run_audio_analysis(path, &spec_config, &extractor, &pool, &audio_postprocessor)
                    .await;
            }
            "6" | "q" | "exit" => {
                println!("  ¡Hasta pronto!");
                break;
            }
            _ => {
                println!("  ❌ Opción no válida. Por favor ingresa un número del 1 al 6.");
            }
        }
    }

    Ok(())
}

async fn run_vision_analysis(
    path: &str,
    preprocessor: &VisionPreprocessor,
    pool: &SessionPool,
    postprocessor: &VisionPostprocessor,
) {
    if !Path::new(path).exists() {
        println!("  ❌ Error: Archivo no encontrado en '{}'", path);
        return;
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("  ❌ Error leyendo archivo: {}", e);
            return;
        }
    };

    println!(
        "\n  🖼️ Analizando imagen: '{}' ({} bytes)...",
        path,
        bytes.len()
    );
    let t0 = Instant::now();
    let (tensor, (orig_w, orig_h), _letterbox) = match preprocessor.process_bytes(&bytes) {
        Ok(res) => res,
        Err(e) => {
            println!("  ❌ Error decodificando imagen: {}", e);
            return;
        }
    };
    let t_pre = t0.elapsed();

    let shape = vec![
        1,
        3,
        preprocessor.target_h as usize,
        preprocessor.target_w as usize,
    ];
    let t1 = Instant::now();
    let (logits, _us) = match pool.run(ModelKind::Vision, &shape, tensor).await {
        Ok(res) => res,
        Err(e) => {
            println!("  ❌ Error en inferencia ONNX: {}", e);
            return;
        }
    };
    let t_inf = t1.elapsed();

    let t2 = Instant::now();
    let probs = VisionPostprocessor::softmax(&logits);
    let preds = postprocessor.extract_top_k(&probs, 3);
    let t_post = t2.elapsed();
    let t_total = t_pre + t_inf + t_post;

    println!("  ------------------------------------------------------------------");
    println!(
        "  📐 Dimensiones originales: {}x{} px -> Letterbox a 224x224",
        orig_w, orig_h
    );
    println!("  🎯 PREDICCIONES DEL MODELO (Top-3):");
    for (i, p) in preds.iter().enumerate() {
        let bar_len = (p.probability * 30.0).round() as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "     [{}] {:<22} {:>5.1}% | {:<30}",
            i + 1,
            p.label,
            p.probability * 100.0,
            bar
        );
    }
    println!("  ------------------------------------------------------------------");
    println!("  ⏱️ DESGLOSE DE TIEMPOS:");
    println!(
        "     • Preproceso (Decode + Letterbox + NCHW): {:.2} ms",
        t_pre.as_secs_f64() * 1000.0
    );
    println!(
        "     • Inferencia ONNX SIMD:                 {:.2} ms",
        t_inf.as_secs_f64() * 1000.0
    );
    println!(
        "     • Postproceso (Softmax + Top-K):        {:.3} ms",
        t_post.as_secs_f64() * 1000.0
    );
    println!(
        "     🚀 LATENCIA TOTAL:                       {:.2} ms ({:.1} FPS)",
        t_total.as_secs_f64() * 1000.0,
        1.0 / t_total.as_secs_f64()
    );
}

async fn run_audio_analysis(
    path: &str,
    spec_config: &SpectrogramConfig,
    extractor: &MelSpectrogramExtractor,
    pool: &SessionPool,
    audio_postprocessor: &AudioPostprocessor,
) {
    if !Path::new(path).exists() {
        println!("  ❌ Error: Archivo no encontrado en '{}'", path);
        return;
    }

    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("  ❌ Error leyendo archivo: {}", e);
            return;
        }
    };

    println!(
        "\n  🎙️ Analizando archivo WAV: '{}' ({} bytes)...",
        path,
        bytes.len()
    );
    let t0 = Instant::now();
    let (raw_samples, sample_rate) = match WavDecoder::decode_bytes(&bytes) {
        Ok(res) => res,
        Err(e) => {
            println!("  ❌ Error decodificando WAV: {}", e);
            return;
        }
    };
    let resampled = AudioResampler::resample(&raw_samples, sample_rate, spec_config.sample_rate);
    let t_dec = t0.elapsed();

    let t1 = Instant::now();
    let mel_tensor = extractor.extract(&resampled);
    let t_spec = t1.elapsed();

    let shape = vec![1, 1, spec_config.n_mels, spec_config.target_frames];
    let t2 = Instant::now();
    let (logits, _us) = match pool.run(ModelKind::Audio, &shape, mel_tensor).await {
        Ok(res) => res,
        Err(e) => {
            println!("  ❌ Error en inferencia ONNX: {}", e);
            return;
        }
    };
    let t_inf = t2.elapsed();

    let t3 = Instant::now();
    let probs = AudioPostprocessor::softmax(&logits);
    let preds = audio_postprocessor.extract_top_k(&probs, 3);
    let t_post = t3.elapsed();
    let t_total = t_dec + t_spec + t_inf + t_post;

    let dur = raw_samples.len() as f32 / sample_rate as f32;
    println!("  ------------------------------------------------------------------");
    println!(
        "  🔊 Tasa original: {} Hz | Duración: {:.2}s | Resampleado a {} Hz",
        sample_rate, dur, spec_config.sample_rate
    );
    println!("  🗣️ COMANDOS ACÚSTICOS DETECTADOS (Top-3):");
    for (i, p) in preds.iter().enumerate() {
        let bar_len = (p.probability * 30.0).round() as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "     [{}] {:<22} {:>5.1}% | {:<30}",
            i + 1,
            p.keyword,
            p.probability * 100.0,
            bar
        );
    }
    println!("  ------------------------------------------------------------------");
    println!("  ⏱️ DESGLOSE DE TIEMPOS:");
    println!(
        "     • Decodificación + Resample a 16kHz:    {:.2} ms",
        t_dec.as_secs_f64() * 1000.0
    );
    println!(
        "     • Espectrograma Log-Mel (80 bins):       {:.2} ms",
        t_spec.as_secs_f64() * 1000.0
    );
    println!(
        "     • Inferencia ONNX SIMD:                 {:.2} ms",
        t_inf.as_secs_f64() * 1000.0
    );
    println!(
        "     • Postproceso (Softmax + Top-K):        {:.3} ms",
        t_post.as_secs_f64() * 1000.0
    );
    println!(
        "     🚀 LATENCIA TOTAL:                       {:.2} ms ({:.1}x tiempo real)",
        t_total.as_secs_f64() * 1000.0,
        dur as f64 / t_total.as_secs_f64()
    );
}
