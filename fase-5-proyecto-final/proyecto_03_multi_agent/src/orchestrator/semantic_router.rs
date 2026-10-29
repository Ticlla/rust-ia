//! Enrutador Semántico Vectorial (Semantic Router) para Clasificación Robusta de Intenciones.
//!
//! Reemplaza las heurísticas frágiles de expresiones regulares por proyecciones densas
//! en el espacio vectorial y cálculo de similitud coseno con anclajes representativos,
//! resolviendo ambigüedades, negaciones y paráfrasis ("¿qué opinas de Rust?" vs "crea un RingBuffer").

use std::fmt;

/// Intención clasificada del usuario
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum UserIntent {
    /// Diálogo conversacional, opiniones, explicaciones teóricas, saludos
    Conversational,
    /// Requerimiento de ingeniería, diseño de arquitectura o generación de código (activa DAG)
    TechnicalTask,
}

impl fmt::Display for UserIntent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Conversational => write!(f, "Conversational"),
            Self::TechnicalTask => write!(f, "TechnicalTask"),
        }
    }
}

pub const EMBEDDING_DIM: usize = 128;

/// Extractor de representaciones vectoriales densas L2-normalizadas basado en n-gramas de subpalabras (FastText style)
#[derive(Clone)]
pub struct DenseSubwordEmbedder {
    dim: usize,
}

impl Default for DenseSubwordEmbedder {
    fn default() -> Self {
        Self { dim: EMBEDDING_DIM }
    }
}

impl DenseSubwordEmbedder {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    /// FNV-1a Hash determinista de 64 bits
    #[inline(always)]
    fn fnv1a(text: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in text.bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    /// Genera un embedding denso continuo L2-normalizado para una cadena de texto
    pub fn embed(&self, text: &str) -> Vec<f32> {
        let cleaned: String = text
            .to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();

        let words: Vec<&str> = cleaned.split_whitespace().collect();
        if words.is_empty() {
            return vec![0.0f32; self.dim];
        }

        let mut vec = vec![0.0f32; self.dim];
        let mut in_negation = false;

        for w in &words {
            // Detección de negación contextual (Spanish & English)
            if *w == "no"
                || *w == "sin"
                || *w == "not"
                || *w == "never"
                || *w == "dont"
                || *w == "don't"
            {
                in_negation = true;
                continue;
            }

            // Prefijo de token según estado de negación
            let token_key = if in_negation {
                format!("neg_{w}")
            } else {
                w.to_string()
            };

            // 1. Hash de la palabra completa
            let h = Self::fnv1a(&token_key);
            let idx = (h as usize) % self.dim;
            let sign = if (h >> 32) % 2 == 0 { 1.0f32 } else { -1.0f32 };
            vec[idx] += sign * 1.5;

            // 2. Hash de n-gramas de caracteres (subpalabras de longitud 2, 3 y 4)
            let chars: Vec<char> = w.chars().collect();
            for n in [2, 3, 4] {
                if chars.len() >= n {
                    for i in 0..=(chars.len() - n) {
                        let sub: String = chars[i..i + n].iter().collect();
                        let sub_key = if in_negation {
                            format!("neg_{sub}")
                        } else {
                            sub
                        };
                        let sh = Self::fnv1a(&sub_key);
                        let sidx = (sh as usize) % self.dim;
                        let ssign = if (sh >> 32) % 2 == 0 { 1.0f32 } else { -1.0f32 };
                        vec[sidx] += ssign * 0.6;
                    }
                }
            }
        }

        // Normalización L2 unitaria
        let norm_sq: f32 = vec.iter().map(|&x| x * x).sum();
        if norm_sq > 1e-12 {
            let norm = norm_sq.sqrt();
            for x in &mut vec {
                *x /= norm;
            }
        }

        vec
    }

    /// Cálculo de similitud coseno entre dos vectores densos normalizados
    #[inline(always)]
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        // Auto-vectorizable con desenrollado de bucle
        let len = a.len();
        let chunks = len / 4;
        let mut sum0 = 0.0f32;
        let mut sum1 = 0.0f32;
        let mut sum2 = 0.0f32;
        let mut sum3 = 0.0f32;

        for i in 0..chunks {
            let base = i * 4;
            sum0 += a[base] * b[base];
            sum1 += a[base + 1] * b[base + 1];
            sum2 += a[base + 2] * b[base + 2];
            sum3 += a[base + 3] * b[base + 3];
        }

        let mut total = (sum0 + sum1) + (sum2 + sum3);
        let rem_start = chunks * 4;
        for i in rem_start..len {
            total += a[i] * b[i];
        }

        total.clamp(-1.0, 1.0)
    }
}

/// Resultado detallado de la clasificación semántica
#[derive(Debug, Clone)]
pub struct RouteMatch {
    pub intent: UserIntent,
    pub confidence: f32,
    pub top_anchor: String,
    pub conv_similarity: f32,
    pub tech_similarity: f32,
    pub is_ambiguous: bool,
}

/// Enrutador semántico de dos niveles con anclas vectoriales pre-calculadas
#[derive(Clone)]
pub struct SemanticRouter {
    embedder: DenseSubwordEmbedder,
    conv_anchors: Vec<(&'static str, Vec<f32>)>,
    tech_anchors: Vec<(&'static str, Vec<f32>)>,
}

impl Default for SemanticRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl SemanticRouter {
    pub fn new() -> Self {
        let embedder = DenseSubwordEmbedder::default();

        // Anclas semánticas para Ruta Conversacional / Opinión / Explicación
        let conv_texts = [
            "hola",
            "hi",
            "hello",
            "hey",
            "saludos",
            "buenas",
            "qué tal",
            "cómo estás",
            "how are you",
            "quién eres",
            "who are you",
            "qué opinas de este lenguaje o tecnología",
            "cuál es tu opinión sobre la seguridad y concurrencia",
            "qué piensas sobre este tema",
            "te gusta la filosofía de este sistema",
            "dame tu perspectiva y opinión",
            "cuéntame qué opinas de rust",
            "qué opinas de rust vs cpp",
            "qué opinas de python y rust",
            "hola quién eres y qué puedes hacer",
            "buenos días cómo estás",
            "saludos cordiales",
            "explícame qué es el ownership sin código",
            "cuál es el concepto teórico detrás de tokio",
            "no quiero código solo tu opinión",
            "no me des código dime tu perspectiva",
            "muchas gracias por tu ayuda",
            "chao hasta luego",
            "what do you think about rust",
            "what is your opinion on memory safety",
            "who are you and what can you do",
            "hello how are you",
            "thank you very much",
        ];

        // Anclas semánticas para Ruta de Ingeniería Técnica / Construcción / Código (DAG)
        let tech_texts = [
            "diseñar una estructura de datos concurrente",
            "implementar un ringbuffer continuo en rust",
            "crear módulo simd avx2 de producto punto",
            "escribe el código para una caché lru",
            "desarrollar un actor con canales mpsc de tokio",
            "genera el struct y las funciones asociadas en rust",
            "construir un algoritmo de memoria contigua",
            "implementa una cola lock-free",
            "escribir código para evaluar expresiones",
            "optimiza este bloque de código",
            "audita este código buscando unsafe o leaks",
            "make me a hello world program in rust",
            "design a concurrent ringbuffer",
            "build a simd avx2 dot product function",
            "implement an lru cache struct in rust",
            "write rust code for this problem",
        ];

        let conv_anchors = conv_texts.iter().map(|&t| (t, embedder.embed(t))).collect();

        let tech_anchors = tech_texts.iter().map(|&t| (t, embedder.embed(t))).collect();

        Self {
            embedder,
            conv_anchors,
            tech_anchors,
        }
    }

    /// Clasifica una consulta mediante similitud vectorial máxima con anclas precalculadas
    pub fn classify(&self, query: &str) -> RouteMatch {
        let q_trimmed = query.trim().to_lowercase();
        if q_trimmed == "hi"
            || q_trimmed == "hola"
            || q_trimmed == "hello"
            || q_trimmed == "hey"
            || q_trimmed == "saludos"
            || q_trimmed == "buenas"
        {
            return RouteMatch {
                intent: UserIntent::Conversational,
                confidence: 0.99,
                top_anchor: "saludo directo".to_string(),
                conv_similarity: 1.0,
                tech_similarity: 0.0,
                is_ambiguous: false,
            };
        }

        let q_embed = self.embedder.embed(query);

        // 1. Similitud máxima con anclas conversacionales
        let mut max_conv_sim = -1.0f32;
        let mut top_conv_anchor = "";
        for (anchor_text, anchor_vec) in &self.conv_anchors {
            let sim = DenseSubwordEmbedder::cosine_similarity(&q_embed, anchor_vec);
            if sim > max_conv_sim {
                max_conv_sim = sim;
                top_conv_anchor = anchor_text;
            }
        }

        // 2. Similitud máxima con anclas de ingeniería técnica
        let mut max_tech_sim = -1.0f32;
        let mut top_tech_anchor = "";
        for (anchor_text, anchor_vec) in &self.tech_anchors {
            let sim = DenseSubwordEmbedder::cosine_similarity(&q_embed, anchor_vec);
            if sim > max_tech_sim {
                max_tech_sim = sim;
                top_tech_anchor = anchor_text;
            }
        }

        // 3. Resolución de intención con margen de decisión y detección de negación
        let margin = (max_tech_sim - max_conv_sim).abs();
        let is_ambiguous = margin < 0.05;

        // Comprobación de modificador explícito de negación ("no quiero código", "sin código")
        let q_lower = query.to_lowercase();
        let explicitly_denies_code = q_lower.contains("no quiero c")
            || q_lower.contains("sin c")
            || q_lower.contains("no me des c")
            || q_lower.contains("no code")
            || q_lower.contains("without code");

        let (intent, confidence, top_anchor) = if explicitly_denies_code {
            (
                UserIntent::Conversational,
                0.95,
                "negación explícita de código".to_string(),
            )
        } else if max_tech_sim > max_conv_sim + 0.04 {
            (
                UserIntent::TechnicalTask,
                max_tech_sim,
                top_tech_anchor.to_string(),
            )
        } else if max_conv_sim > max_tech_sim + 0.04 {
            (
                UserIntent::Conversational,
                max_conv_sim,
                top_conv_anchor.to_string(),
            )
        } else {
            // Zona frontera: si contiene verbos de acción constructiva imperativa sin negación
            let has_creation_imperative = [
                "crear",
                "crea ",
                "diseñar",
                "diseña ",
                "implementar",
                "implementa ",
                "make ",
                "build ",
            ]
            .iter()
            .any(|&verb| q_lower.contains(verb));

            if has_creation_imperative {
                (
                    UserIntent::TechnicalTask,
                    max_tech_sim.max(0.60),
                    top_tech_anchor.to_string(),
                )
            } else {
                (
                    UserIntent::Conversational,
                    max_conv_sim.max(0.60),
                    top_conv_anchor.to_string(),
                )
            }
        };

        RouteMatch {
            intent,
            confidence,
            top_anchor,
            conv_similarity: max_conv_sim,
            tech_similarity: max_tech_sim,
            is_ambiguous,
        }
    }

    /// Retorna directamente el `UserIntent` para integración transparente
    pub fn classify_intent(&self, query: &str) -> UserIntent {
        self.classify(query).intent
    }
}
