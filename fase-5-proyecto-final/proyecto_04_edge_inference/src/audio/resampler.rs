//! Re-muestreador lineal de audio para normalización de frecuencias de muestreo.

pub struct AudioResampler;

impl AudioResampler {
    /// Remuestrea una señal de audio de una tasa original `from_rate` a una tasa objetivo `to_rate`
    pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        if from_rate == to_rate || samples.is_empty() {
            return samples.to_vec();
        }

        let ratio = from_rate as f64 / to_rate as f64;
        let new_len = ((samples.len() as f64) / ratio).round() as usize;
        let mut output = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let orig_pos = i as f64 * ratio;
            let idx0 = orig_pos.floor() as usize;
            let idx1 = (idx0 + 1).min(samples.len() - 1);
            let frac = (orig_pos - idx0 as f64) as f32;

            let s0 = samples[idx0];
            let s1 = samples[idx1];
            let interpolated = s0 * (1.0 - frac) + s1 * frac;
            output.push(interpolated);
        }

        output
    }
}
