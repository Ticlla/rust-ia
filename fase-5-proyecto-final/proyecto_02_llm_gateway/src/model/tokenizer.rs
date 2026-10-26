//! Tokenizador para el Gateway de Inferencia

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct GatewayTokenizer {
    pub inner: Tokenizer,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
}

impl GatewayTokenizer {
    pub fn new_builtin() -> Self {
        let vocab = [
            ("<unk>".to_string(), 0),
            ("<s>".to_string(), 1),
            ("</s>".to_string(), 2),
            ("<pad>".to_string(), 3),
            ("Hola".to_string(), 4),
            ("¿Cómo".to_string(), 5),
            ("estás?".to_string(), 6),
            ("El".to_string(), 7),
            ("futuro".to_string(), 8),
            ("de".to_string(), 9),
            ("la".to_string(), 10),
            ("IA".to_string(), 11),
            ("en".to_string(), 12),
            ("Rust".to_string(), 13),
            ("es".to_string(), 14),
            ("rápido,".to_string(), 15),
            ("seguro".to_string(), 16),
            ("y".to_string(), 17),
            ("escalable.".to_string(), 18),
            ("vLLM".to_string(), 19),
            ("sirve".to_string(), 20),
            ("modelos".to_string(), 21),
            ("con".to_string(), 22),
            ("Continuous".to_string(), 23),
            ("Batching.".to_string(), 24),
            ("Respuesta:".to_string(), 25),
            ("Excelente".to_string(), 26),
            ("pregunta.".to_string(), 27),
            ("Procesando".to_string(), 28),
            ("consulta".to_string(), 29),
        ];

        let wl = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("<unk>".to_string())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(wl);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

        Self {
            inner: tokenizer,
            eos_token_id: 2,
            pad_token_id: 3,
        }
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| e.to_string())?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.inner.decode(ids, false).map_err(|e| e.to_string())
    }
}
