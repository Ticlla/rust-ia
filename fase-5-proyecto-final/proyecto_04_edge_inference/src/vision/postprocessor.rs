//! Postprocesador de inferencia de visión: Softmax numéricamente estable y extracción Top-K.

use crate::vision::types::VisionClassPrediction;

pub struct VisionPostprocessor {
    pub class_labels: Vec<String>,
}

impl Default for VisionPostprocessor {
    fn default() -> Self {
        Self {
            class_labels: vec![
                "inspeccion_conforme".to_string(),
                "defecto_superficial".to_string(),
                "grieta_estructural".to_string(),
                "etiqueta_desalineada".to_string(),
                "anomalia_termica".to_string(),
            ],
        }
    }
}

impl VisionPostprocessor {
    pub fn new(labels: Vec<String>) -> Self {
        Self {
            class_labels: labels,
        }
    }

    /// Softmax numéricamente estable para evitar desbordamiento aritmético
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

    /// Extrae las Top-K predicciones ordenadas por confianza descendente
    pub fn extract_top_k(&self, probabilities: &[f32], k: usize) -> Vec<VisionClassPrediction> {
        let mut indexed: Vec<(usize, f32)> = probabilities
            .iter()
            .cloned()
            .enumerate()
            .collect();

        // Ordenamiento descendente por probabilidad
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_count = k.min(indexed.len());
        indexed
            .into_iter()
            .take(top_count)
            .map(|(id, prob)| {
                let label = self
                    .class_labels
                    .get(id)
                    .cloned()
                    .unwrap_or_else(|| format!("clase_{id}"));
                VisionClassPrediction {
                    class_id: id,
                    label,
                    probability: prob,
                }
            })
            .collect()
    }
}
