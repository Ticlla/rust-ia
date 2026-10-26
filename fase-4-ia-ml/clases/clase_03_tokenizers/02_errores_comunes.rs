//! ==============================================================================
//! CLASE 03: Errores Comunes en Tokenización para LLMs
//! ==============================================================================
//!
//! En el pre-procesamiento de texto para modelos de lenguaje, pequeños errores
//! de tokenización pueden corromper la atención o hacer crashear la red neuronal.

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones en Tokenización");
    println!("============================================================\n");

    let tokenizer = crear_tokenizer_base()?;

    error_1_desbordamiento_sin_truncamiento(&tokenizer)?;
    error_2_ignorar_el_attention_mask(&tokenizer)?;
    error_3_decodificar_con_tokens_especiales(&tokenizer)?;

    println!("\n✅ Todos los antipatrones analizados y prevenidos correctamente.");
    Ok(())
}

fn crear_tokenizer_base() -> Result<Tokenizer, Error> {
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
    let _ = tokenizer.add_special_tokens([tokenizers::AddedToken::from("<pad>", true)]);
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
    Ok(tokenizer)
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Desbordamiento de la Ventana de Contexto (Sin Truncamiento)
/// ----------------------------------------------------------------------------
/// Si un usuario envía un documento de 8,000 tokens a un modelo con contexto
/// de 4,096, la red neuronal intentará acceder a matrices posicionales que no existen.
fn error_1_desbordamiento_sin_truncamiento(tokenizer: &Tokenizer) -> Result<(), Error> {
    println!("--- Error 1: Desbordamiento de Contexto sin Truncamiento ---");

    let mut tok = tokenizer.clone();
    // Configuramos truncamiento estricto a un máximo de 2 tokens:
    tok.with_truncation(Some(TruncationParams {
        max_length: 2,
        ..Default::default()
    }))?;

    let texto_largo = "rust ia ru st"; // 4 tokens
    let encoding = tok.encode(texto_largo, true)?;

    println!("Texto original (4 tokens): \"{}\"", texto_largo);
    println!("✓ Truncado automáticamente a longitud máxima: {:?}", encoding.get_tokens());
    assert_eq!(encoding.get_ids().len(), 2);
    println!("💡 Regla: Siempre fija `with_truncation` igual a la ventana de contexto de tu LLM.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Ignorar el `attention_mask` en Lotes con Relleno
/// ----------------------------------------------------------------------------
/// Si la frase tiene padding `[11, 10, 0, 0]` y no le pasas el attention_mask
/// `[1, 1, 0, 0]` al Transformer, las cabezas de atención calcularán probabilidades
/// sobre tokens vacíos, destruyendo la coherencia de la respuesta.
fn error_2_ignorar_el_attention_mask(tokenizer: &Tokenizer) -> Result<(), Error> {
    println!("--- Error 2: Ignorar el Attention Mask en Inferencia ---");

    let mut tok = tokenizer.clone();
    tok.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(4),
        pad_id: 0,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));

    let encoding = tok.encode("rust ia", true)?; // 2 tokens reales + 2 pads

    println!("Tokens generados con relleno: {:?}", encoding.get_tokens());
    println!("Attention Mask:               {:?}", encoding.get_attention_mask());
    println!("💡 Regla: El Transformer solo debe prestar atención donde mask == 1.\n");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Mostrar Tokens Especiales en la Salida al Usuario
/// ----------------------------------------------------------------------------
/// Si devuelves `tokenizer.decode(&ids, false)`, tokens internos como `<pad>`,
/// `[CLS]` o `<|im_end|>` aparecerán impresos en la respuesta para el usuario.
fn error_3_decodificar_con_tokens_especiales(tokenizer: &Tokenizer) -> Result<(), Error> {
    println!("--- Error 3: Tokens Especiales Visibles al Usuario ---");

    let ids_con_pad = vec![11, 10, 0, 0]; // "rust ia <pad> <pad>"

    let sucio = tokenizer.decode(&ids_con_pad, false)?;
    let limpio = tokenizer.decode(&ids_con_pad, true)?;

    println!("❌ Con skip_special_tokens=false: \"{}\"", sucio);
    println!("✓ Con skip_special_tokens=true:  \"{}\"", limpio);
    Ok(())
}
