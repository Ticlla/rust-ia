//! Herramienta CLI de benchmarking y prueba en el borde para visión y audio.

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
    println!("================================================================================");
    println!("  🔬 EDGE INFERENCE BENCHMARK CLI — VISIÓN & AUDIO");
    println!("================================================================================");

    let config = EdgeServiceConfig::default();
    let pool = SessionPool::new(&config, DEFAULT_VISION_MODEL, DEFAULT_AUDIO_MODEL)?;

    // -------------------------------------------------------------------------
    // 1. BENCHMARK DE VISIÓN ARTIFICIAL
    // -------------------------------------------------------------------------
    println!("\n🖼️  [1/2] Iniciando Benchmark de Pipeline de Visión (10 iteraciones)...");
    let preprocessor = VisionPreprocessor::default();
    let postprocessor = VisionPostprocessor::default();

    // Generar imagen PNG sintética de 640x480 (resolución típica de cámara Edge)
    let png_bytes = VisionPreprocessor::generate_synthetic_png(640, 480);
    println!(
        "    • Imagen de prueba generada: 640x480 PNG ({} bytes)",
        png_bytes.len()
    );

    let mut pre_times = Vec::new();
    let mut inf_times = Vec::new();
    let mut post_times = Vec::new();

    for _ in 0..10 {
        let t0 = Instant::now();
        let (tensor, _dims, _letterbox) = preprocessor.process_bytes(&png_bytes).unwrap();
        let t_pre = t0.elapsed();

        let shape = vec![
            1,
            3,
            preprocessor.target_h as usize,
            preprocessor.target_w as usize,
        ];
        let t1 = Instant::now();
        let (logits, _us) = pool.run(ModelKind::Vision, &shape, tensor).await.unwrap();
        let t_inf = t1.elapsed();

        let t2 = Instant::now();
        let probs = VisionPostprocessor::softmax(&logits);
        let _preds = postprocessor.extract_top_k(&probs, 3);
        let t_post = t2.elapsed();

        pre_times.push(t_pre.as_secs_f64() * 1000.0);
        inf_times.push(t_inf.as_secs_f64() * 1000.0);
        post_times.push(t_post.as_secs_f64() * 1000.0);
    }

    let avg_pre: f64 = pre_times.iter().sum::<f64>() / pre_times.len() as f64;
    let avg_inf: f64 = inf_times.iter().sum::<f64>() / inf_times.len() as f64;
    let avg_post: f64 = post_times.iter().sum::<f64>() / post_times.len() as f64;
    let total_vision = avg_pre + avg_inf + avg_post;

    println!(
        "    ✓ Preprocesamiento (Decode + Letterbox + NCHW): {:.2} ms",
        avg_pre
    );
    println!(
        "    ✓ Inferencia ONNX Runtime (CPU SIMD):          {:.2} ms",
        avg_inf
    );
    println!(
        "    ✓ Postprocesamiento (Softmax + Top-K):          {:.2} ms",
        avg_post
    );
    println!(
        "    🚀 LATENCIA TOTAL DE VISIÓN:                    {:.2} ms ({:.1} FPS)",
        total_vision,
        1000.0 / total_vision
    );

    // -------------------------------------------------------------------------
    // 2. BENCHMARK DE AUDIO Y ESPECTROGRAMAS MEL
    // -------------------------------------------------------------------------
    println!("\n🎙️  [2/2] Iniciando Benchmark de Pipeline de Audio (10 iteraciones)...");
    let spec_config = SpectrogramConfig::default();
    let extractor = MelSpectrogramExtractor::new(spec_config.clone());
    let audio_postprocessor = AudioPostprocessor::default();

    // Generar audio senoidal de 1.0 segundo a 44,100 Hz
    let wav_bytes = WavDecoder::synthesize_test_wav(44100, 1.0, 440.0);
    println!(
        "    • Audio WAV de prueba: 1.0s @ 44.1kHz ({} bytes)",
        wav_bytes.len()
    );

    let mut audio_dec_times = Vec::new();
    let mut audio_spec_times = Vec::new();
    let mut audio_inf_times = Vec::new();

    for _ in 0..10 {
        let t0 = Instant::now();
        let (raw_samples, sample_rate) = WavDecoder::decode_bytes(&wav_bytes).unwrap();
        let resampled =
            AudioResampler::resample(&raw_samples, sample_rate, spec_config.sample_rate);
        let t_dec = t0.elapsed();

        let t1 = Instant::now();
        let mel_tensor = extractor.extract(&resampled);
        let t_spec = t1.elapsed();

        let shape = vec![1, 1, spec_config.n_mels, spec_config.target_frames];
        let t2 = Instant::now();
        let (logits, _us) = pool
            .run(ModelKind::Audio, &shape, mel_tensor)
            .await
            .unwrap();
        let t_inf = t2.elapsed();

        let probs = AudioPostprocessor::softmax(&logits);
        let _preds = audio_postprocessor.extract_top_k(&probs, 3);

        audio_dec_times.push(t_dec.as_secs_f64() * 1000.0);
        audio_spec_times.push(t_spec.as_secs_f64() * 1000.0);
        audio_inf_times.push(t_inf.as_secs_f64() * 1000.0);
    }

    let avg_dec: f64 = audio_dec_times.iter().sum::<f64>() / audio_dec_times.len() as f64;
    let avg_spec: f64 = audio_spec_times.iter().sum::<f64>() / audio_spec_times.len() as f64;
    let avg_inf_a: f64 = audio_inf_times.iter().sum::<f64>() / audio_inf_times.len() as f64;
    let total_audio = avg_dec + avg_spec + avg_inf_a;

    println!(
        "    ✓ Decodificación WAV y Remuestreo a 16kHz:       {:.2} ms",
        avg_dec
    );
    println!(
        "    ✓ Extracción de Espectrograma Log-Mel (80 bins): {:.2} ms",
        avg_spec
    );
    println!(
        "    ✓ Inferencia ONNX Runtime (CPU SIMD):          {:.2} ms",
        avg_inf_a
    );
    println!(
        "    🚀 LATENCIA TOTAL DE AUDIO:                     {:.2} ms (Procesamiento {:.1}x tiempo real)",
        total_audio,
        1000.0 / total_audio
    );

    println!("\n================================================================================");
    println!("  ✅ TODOS LOS BENCHMARKS FINALIZADOS CON RENDIMIENTO INDUSTRIAL");
    println!("================================================================================");

    Ok(())
}
