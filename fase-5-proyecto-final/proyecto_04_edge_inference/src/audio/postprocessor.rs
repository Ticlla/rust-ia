//! Postprocesador de audio: Softmax y extracción de comandos de voz / palabras clave (KWS).

use crate::audio::types::AudioKeywordPrediction;

pub struct AudioPostprocessor {
    pub keyword_labels: Vec<String>,
}

impl Default for AudioPostprocessor {
    fn default() -> Self {
        Self {
            keyword_labels: vec![
                "iniciar_proceso".to_string(),
                "parada_emergencia".to_string(),
                "aumentar_velocidad".to_string(),
                "reducir_velocidad".to_string(),
                "ruido_fondo".to_string(),
            ],
        }
    }
}

impl AudioPostprocessor {
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            keyword_labels: labels,
        }
    }

    /// Softmax numéricamente estable
    pub fn softmax(logits: &[f32]) -> Vec<f32> {
        if logits.is_empty() {
            return Vec::new();
        }

        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
        let sum_exps: f32 = exps.iter().sum();

        if sum_exps == 0.0 {
            return vec![1.0 / logits.len() as f32; logits.len()];
        }

        exps.into_iter().map(|v| v / sum_exps).collect()
    }

    /// Extrae las Top-K palabras clave ordenadas por confianza
    pub fn extract_top_k(&self, probabilities: &[f32], k: usize) -> Vec<AudioKeywordPrediction> {
        let mut indexed: Vec<(usize, f32)> = probabilities
            .iter()
            .cloned()
            .enumerate()
            .collect();

        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_count = k.min(indexed.len());
        indexed
            .into_iter()
            .take(top_count)
            .map(|(id, prob)| {
                let keyword = self
                    .keyword_labels
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("comando_{id}"));
                AudioKeywordPrediction {
                    keyword_id: id,
                    keyword,
                    probability: prob,
                }
            })
            .collect()
    }
}
