//! Extractor de espectrogramas Mel (Log-Mel Filterbanks) en Rust puro.

use crate::audio::types::SpectrogramConfig;

pub struct MelSpectrogramExtractor {
    pub config: SpectrogramConfig,
    hann_window: Vec<f32>,
    mel_filterbank: Vec<Vec<f32>>, // [n_mels][num_freq_bins]
    num_freq_bins: usize,
}

impl MelSpectrogramExtractor {
    pub fn new(config: SpectrogramConfig) -> Self {
        let n_fft = config.n_fft;
        let num_freq_bins = n_fft / 2 + 1;

        // 1. Precomputar Ventana de Hann
        let mut hann_window = Vec::with_capacity(n_fft);
        for n in 0..n_fft {
            let val = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / (n_fft - 1) as f32).cos());
            hann_window.push(val);
        }

        // 2. Precomputar Banco de Filtros Mel triangulares
        let mel_filterbank = Self::build_mel_filterbank(
            config.n_mels,
            num_freq_bins,
            config.sample_rate,
            0.0,
            (config.sample_rate as f32) / 2.0,
        );

        Self {
            config,
            hann_window,
            mel_filterbank,
            num_freq_bins,
        }
    }

    /// Construye matriz de bancos triangulares Mel espaciados uniformemente en escala Mel
    fn build_mel_filterbank(
        n_mels: usize,
        num_freq_bins: usize,
        sample_rate: u32,
        f_min: f32,
        f_max: f32,
    ) -> Vec<Vec<f32>> {
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        // n_mels + 2 puntos para definir los vértices de los triángulos
        let mut mel_points = Vec::with_capacity(n_mels + 2);
        let step = (mel_max - mel_min) / (n_mels + 1) as f32;
        for i in 0..=(n_mels + 1) {
            mel_points.push(mel_to_hz(mel_min + (i as f32) * step));
        }

        // Convertir Hz a índices discretos de Fourier
        let bin_width = (sample_rate as f32) / (2.0 * (num_freq_bins - 1) as f32);
        let bin_indices: Vec<f32> = mel_points.into_iter().map(|hz| hz / bin_width).collect();

        let mut filterbank = vec![vec![0.0f32; num_freq_bins]; n_mels];

        for m in 0..n_mels {
            let left = bin_indices[m];
            let center = bin_indices[m + 1];
            let right = bin_indices[m + 2];

            for k in 0..num_freq_bins {
                let k_f = k as f32;
                if k_f >= left && k_f <= center && (center - left) > 0.0 {
                    filterbank[m][k] = (k_f - left) / (center - left);
                } else if k_f > center && k_f <= right && (right - center) > 0.0 {
                    filterbank[m][k] = (right - k_f) / (right - center);
                }
            }
        }

        filterbank
    }

    /// Extrae el espectrograma Log-Mel completo para una señal de audio mono
    /// Retorna un buffer plano NCHW de dimensiones [1, 1, n_mels, target_frames]
    pub fn extract(&self, samples: &[f32]) -> Vec<f32> {
        let n_fft = self.config.n_fft;
        let hop = self.config.hop_length;
        let n_mels = self.config.n_mels;
        let target_frames = self.config.target_frames;

        // Calcular número de tramas disponibles
        let available_frames = if samples.len() >= n_fft {
            (samples.len() - n_fft) / hop + 1
        } else {
            1
        };

        let frames_to_process = available_frames.min(target_frames);

        // Matriz intermedia: [n_mels][target_frames]
        let mut mel_spectrogram = vec![vec![0.0f32; target_frames]; n_mels];

        // Buffer temporal para aplicar ventana
        let mut windowed = vec![0.0f32; n_fft];

        for frame_idx in 0..frames_to_process {
            let start = frame_idx * hop;

            // Extraer y ventanear la trama
            for n in 0..n_fft {
                let sample_val = if start + n < samples.len() {
                    samples[start + n]
                } else {
                    0.0
                };
                windowed[n] = sample_val * self.hann_window[n];
            }

            // Calcular densidad espectral de potencia |X[k]|^2
            let mut power_spectrum = vec![0.0f32; self.num_freq_bins];
            for k in 0..self.num_freq_bins {
                let mut re = 0.0f32;
                let mut im = 0.0f32;
                let freq_factor = 2.0 * std::f32::consts::PI * (k as f32) / (n_fft as f32);

                for n in 0..n_fft {
                    let angle = freq_factor * (n as f32);
                    let sample = windowed[n];
                    re += sample * angle.cos();
                    im -= sample * angle.sin();
                }

                power_spectrum[k] = (re * re + im * im) / (n_fft as f32);
            }

            // Multiplicación con bancos Mel y compresión logarítmica
            for m in 0..n_mels {
                let mut energy = 0.0f32;
                let filter = &self.mel_filterbank[m];

                for k in 0..self.num_freq_bins {
                    energy += power_spectrum[k] * filter[k];
                }

                // Compresión Log-Mel: log(max(energy, 1e-6))
                mel_spectrogram[m][frame_idx] = (energy.max(1e-6)).ln();
            }
        }

        // Aplanar en orden continuo [Batch=1, Channel=1, MelBins, TimeFrames]
        let mut flat_tensor = Vec::with_capacity(1 * 1 * n_mels * target_frames);
        for m in 0..n_mels {
            for t in 0..target_frames {
                flat_tensor.push(mel_spectrogram[m][t]);
            }
        }

        flat_tensor
    }
}
