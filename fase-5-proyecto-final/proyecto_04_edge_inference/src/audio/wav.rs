//! Decodificador y sintetizador de audio WAV PCM de alto rendimiento.

use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use std::io::Cursor;

pub struct WavDecoder;

impl WavDecoder {
    /// Decodifica un buffer de bytes WAV a muestras de audio mono en formato float32 [-1.0, 1.0]
    pub fn decode_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
        let cursor = Cursor::new(bytes);
        let reader =
            WavReader::new(cursor).map_err(|e| format!("Fallo leyendo WAV header: {e}"))?;

        let spec = reader.spec();
        let channels = spec.channels as usize;
        let sample_rate = spec.sample_rate;

        if channels == 0 {
            return Err("El archivo WAV no contiene canales de audio".to_string());
        }

        let raw_samples: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
            (SampleFormat::Int, 16) => reader
                .into_samples::<i16>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 32768.0)
                .collect(),
            (SampleFormat::Int, 24) => reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 8388608.0)
                .collect(),
            (SampleFormat::Int, 32) => reader
                .into_samples::<i32>()
                .filter_map(|s| s.ok())
                .map(|s| s as f32 / 2147483648.0)
                .collect(),
            (SampleFormat::Float, 32) => reader
                .into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect(),
            (fmt, bits) => {
                return Err(format!(
                    "Formato de muestra WAV no soportado: formato={fmt:?}, bits={bits}"
                ));
            }
        };

        // Si es multicanal (estéreo), mezcla a mono mediante promediado directo
        let mono_samples = if channels == 1 {
            raw_samples
        } else {
            raw_samples
                .chunks_exact(channels)
                .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
                .collect()
        };

        Ok((mono_samples, sample_rate))
    }

    /// Sintetiza un archivo WAV en memoria con un tono senoidal puro para pruebas
    pub fn synthesize_test_wav(sample_rate: u32, duration_secs: f32, freq_hz: f32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };

        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = WavWriter::new(&mut cursor, spec).unwrap();
            let total_samples = (sample_rate as f32 * duration_secs) as usize;

            for n in 0..total_samples {
                let t = n as f32 / sample_rate as f32;
                let sample_val = (2.0 * std::f32::consts::PI * freq_hz * t).sin();
                let sample_i16 = (sample_val * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer.write_sample(sample_i16).unwrap();
            }
            writer.finalize().unwrap();
        }

        cursor.into_inner()
    }
}
