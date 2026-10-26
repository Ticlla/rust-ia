//! ==============================================================================
//! CLASE 03: Ejercicios Prácticos de Tokenización con Hugging Face
//! ==============================================================================
//!
//! En este laboratorio implementarás y validarás:
//! 1. `construir_bpe_tokenizer`: Crear un tokenizador con vocabulario y merges.
//! 2. `tokenizar_con_offsets`: Extraer IDs y offsets para búsqueda y RAG.
//! 3. `tokenizar_lote_con_padding`: Procesar múltiples textos con atención alineada.

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{Encoding, PaddingParams, PaddingStrategy, Tokenizer};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 03: Tokenización de Alto Rendimiento");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c03_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Construir un Tokenizador BPE
/// ----------------------------------------------------------------------------
pub fn construir_tokenizer_bpe() -> Result<Tokenizer, Error> {
    let vocab: [(String, u32); 12] = [
        ("<pad>".to_string(), 0),
        ("<unk>".to_string(), 1),
        ("r".to_string(), 2),
        ("u".to_string(), 3),
        ("s".to_string(), 4),
        ("t".to_string(), 5),
        ("i".to_string(), 6),
        ("a".to_string(), 7),
        ("ru".to_string(), 8),
        ("st".to_string(), 9),
        ("ia".to_string(), 10),
        ("rust".to_string(), 11),
    ];

    let merges = vec![
        ("r".to_string(), "u".to_string()),
        ("s".to_string(), "t".to_string()),
        ("ru".to_string(), "st".to_string()),
        ("i".to_string(), "a".to_string()),
    ];

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("<unk>".to_string())
        .build()?;

    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
    Ok(tokenizer)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Tokenizar con Offsets y Decodificar
/// ----------------------------------------------------------------------------
pub fn tokenizar_y_decodificar(
    tokenizer: &Tokenizer,
    texto: &str,
) -> Result<(Vec<u32>, String), Error> {
    let encoding = tokenizer.encode(texto, true)?;
    let ids = encoding.get_ids().to_vec();
    let texto_reconstruido = tokenizer.decode(&ids, true)?;
    Ok((ids, texto_reconstruido))
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Tokenización por Lotes con Padding Dinámico
/// ----------------------------------------------------------------------------
pub fn tokenizar_lote(
    tokenizer: &Tokenizer,
    frases: Vec<&str>,
) -> Result<Vec<Encoding>, Error> {
    let mut tok = tokenizer.clone();
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: 0,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));

    let encodings = tok.encode_batch(frases, true)?;
    Ok(encodings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizacion_y_reconstruccion_roundtrip() -> Result<(), Error> {
        let tokenizer = construir_tokenizer_bpe()?;
        let entrada = "rust ia";

        let (ids, texto_recuperado) = tokenizar_y_decodificar(&tokenizer, entrada)?;

        // "rust" -> ID 11, "ia" -> ID 10
        assert_eq!(ids, vec![11, 10]);
        assert_eq!(texto_recuperado, entrada);
        Ok(())
    }

    #[test]
    fn test_subword_merging_logic() -> Result<(), Error> {
        let tokenizer = construir_tokenizer_bpe()?;

        // Caso 1: "rust" se fusiona completamente en 1 token (ID 11)
        let enc1 = tokenizer.encode("rust", true)?;
        assert_eq!(enc1.get_tokens(), vec!["rust"]);
        assert_eq!(enc1.get_ids(), vec![11]);

        // Caso 2: "ru st" como están separados por espacio, se quedan como 2 tokens
        let enc2 = tokenizer.encode("ru st", true)?;
        assert_eq!(enc2.get_tokens(), vec!["ru", "st"]);
        assert_eq!(enc2.get_ids(), vec![8, 9]);
        Ok(())
    }

    #[test]
    fn test_batch_tokenization_attention_mask() -> Result<(), Error> {
        let tokenizer = construir_tokenizer_bpe()?;

        let lote = vec![
            "rust",       // 1 token
            "rust ia",    // 2 tokens (el más largo del lote)
        ];

        let encodings = tokenizar_lote(&tokenizer, lote)?;
        assert_eq!(encodings.len(), 2);

        // Ambas secuencias deben estar alineadas a longitud 2
        assert_eq!(encodings[0].get_ids().len(), 2);
        assert_eq!(encodings[1].get_ids().len(), 2);

        // Frase 0: [11, 0] -> Token real + Pad
        assert_eq!(encodings[0].get_ids(), vec![11, 0]);
        assert_eq!(encodings[0].get_attention_mask(), vec![1, 0]);

        // Frase 1: [11, 10] -> Dos tokens reales
        assert_eq!(encodings[1].get_ids(), vec![11, 10]);
        assert_eq!(encodings[1].get_attention_mask(), vec![1, 1]);
        Ok(())
    }
}
