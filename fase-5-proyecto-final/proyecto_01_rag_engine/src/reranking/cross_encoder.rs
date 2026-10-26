//! Reranker Neuronal Cross-Encoder para Evaluación de Relevancia Fina

use candle_core::{Device, Result, Tensor};
use candle_nn::{Linear, Module};

#[derive(Debug, Clone)]
pub struct RerankedCandidate {
    pub chunk_id: String,
    pub relevance_score: f32,
}

pub struct CrossEncoderReranker {
    pub projection: Linear,
    pub scoring_head: Linear,
    pub dev: Device,
}

impl CrossEncoderReranker {
    pub fn new(dev: &Device) -> Result<Self> {
        let hidden_dim = 32;
        let w_proj = Tensor::randn(0.0f32, 0.5f32, (hidden_dim, 4), dev)?; // 4 features cruzadas
        let projection = Linear::new(w_proj, None);

        let w_head = Tensor::randn(0.0f32, 0.5f32, (1, hidden_dim), dev)?;
        let scoring_head = Linear::new(w_head, None);

        Ok(Self {
            projection,
            scoring_head,
            dev: dev.clone(),
        })
    }

    /// Extrae características de interacción cruzada entre la consulta y el texto candidato
    fn extract_interaction_features(query: &str, candidate_text: &str, base_rrf: f32) -> Vec<f32> {
        let q_words: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();
        let c_words: Vec<String> = candidate_text
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        if q_words.is_empty() || c_words.is_empty() {
            return vec![0.0, 0.0, base_rrf, 0.0];
        }

        // 1. Cobertura de términos de la consulta en el candidato
        let mut matches = 0;
        for qw in &q_words {
            if c_words.contains(qw) {
                matches += 1;
            }
        }
        let query_coverage = (matches as f32) / (q_words.len() as f32);

        // 2. Densidad de coincidencia en el texto candidato
        let candidate_density = (matches as f32) / (c_words.len().min(100) as f32);

        // 3. Score base de RRF normalizado
        let rrf_feat = base_rrf * 100.0;

        // 4. Longitud relativa
        let length_ratio = (q_words.len() as f32) / (c_words.len() as f32).max(1.0);

        vec![query_coverage, candidate_density, rrf_feat, length_ratio]
    }

    /// Reordena una lista de candidatos evaluando la interacción profunda con la consulta
    pub fn rerank(
        &self,
        query: &str,
        candidates: &[(String, String, f32)], // (chunk_id, chunk_text, initial_rrf_score)
        top_k: usize,
    ) -> Result<Vec<RerankedCandidate>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut scored = Vec::with_capacity(candidates.len());

        for (chunk_id, text, base_rrf) in candidates {
            let features = Self::extract_interaction_features(query, text, *base_rrf);
            let feat_tensor = Tensor::from_vec(features, (1, 4), &self.dev)?;

            let h = self.projection.forward(&feat_tensor)?;
            let h_act = candle_nn::ops::silu(&h)?;
            let logit = self.scoring_head.forward(&h_act)?;

            // Sigmoid: 1 / (1 + exp(-x)) para obtener score de probabilidad entre 0.0 y 1.0
            let logit_val = logit.flatten_all()?.to_vec1::<f32>()?[0];
            let sigmoid = (1.0 / (1.0 + (-logit_val).exp())) as f32;

            // Score final combinado ponderado
            let final_score = 0.5 * sigmoid + 0.5 * (*base_rrf * 50.0).min(1.0);

            scored.push(RerankedCandidate {
                chunk_id: chunk_id.clone(),
                relevance_score: final_score,
            });
        }

        scored.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        scored.truncate(top_k);
        Ok(scored)
    }
}
