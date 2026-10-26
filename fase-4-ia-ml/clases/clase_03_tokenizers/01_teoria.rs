//! ==============================================================================
//! CLASE 03: Tokenización de Alto Rendimiento con `tokenizers` de Hugging Face
//! ==============================================================================
//!
//! Los modelos de lenguaje (LLMs como LLaMA, GPT-4, Qwen y BERT) no pueden leer
//! texto directamente. La tokenización es el paso crítico que traduce el lenguaje
//! humano a tensores de números enteros (`u32`) procesables por la red.
//!
//! La famosa librería `tokenizers` de Hugging Face en Python en realidad está
//! escrita 100% en Rust. En esta clase aprendemos a usarla de forma nativa.
//!
//! CONTENIDO:
//! 1. Algoritmo BPE (Byte-Pair Encoding): Vocabulario base y reglas de fusión (Merges).
//! 2. Codificación (`encode`): Extracción de IDs numéricos y Offsets de caracteres.
//! 3. Decodificación (`decode`): Reconstrucción del texto original.
//! 4. Padding y Attention Mask: Alineación de lotes para inferencia paralela.
//! 5. Persistencia: Exportar y cargar desde archivos `tokenizer.json` estándar.

use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer};

type Error = Box<dyn std::error::Error + Send + Sync>;

fn main() -> Result<(), Error> {
    println!("============================================================");
    println!("🔤 CLASE 03: TOKENIZACIÓN DE ALTO RENDIMIENTO (Hugging Face)");
    println!("============================================================\n");

    println!("--- 1. Construcción del Vocabulario BPE y Reglas de Merge ---");
    // Definimos un vocabulario base de caracteres y subpalabras:
    let vocab: [(String, u32); 12] = [
        ("<pad>".to_string(), 0), // Token de relleno (Padding)
        ("<unk>".to_string(), 1), // Token desconocido (Unknown)
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

    // Reglas de fusión (Merges): Indican qué subpalabras se unen primero
    let merges = vec![
        ("r".to_string(), "u".to_string()),     // r + u -> ru
        ("s".to_string(), "t".to_string()),     // s + t -> st
        ("ru".to_string(), "st".to_string()),   // ru + st -> rust
        ("i".to_string(), "a".to_string()),     // i + a -> ia
    ];

    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .unk_token("<unk>".to_string())
        .build()?;

    let mut tokenizer = Tokenizer::new(bpe);
    // Pre-tokenizador: divide primero por espacios en blanco
    tokenizer.with_pre_tokenizer(Some(Whitespace::default()));

    println!("✓ Tokenizador BPE configurado con {} tokens en vocabulario.\n", tokenizer.get_vocab_size(true));

    println!("--- 2. Codificación: De Texto a Tensores Numéricos ---");
    let texto = "rust ia ru st";
    let encoding = tokenizer.encode(texto, true)?;

    println!("Texto de entrada: \"{}\"", texto);
    println!("Tokens (subpalabras): {:?}", encoding.get_tokens());
    println!("IDs numéricos (u32):   {:?}", encoding.get_ids());
    println!("Offsets en caracteres: {:?}", encoding.get_offsets());

    println!("\n--- 3. Decodificación: Reconstrucción del Texto Original ---");
    let ids = encoding.get_ids();
    let texto_recuperado = tokenizer.decode(ids, true)?;
    println!("Texto decodificado: \"{}\"", texto_recuperado);

    println!("\n--- 4. Padding y Attention Mask para Lotes (Batching) ---");
    // En inferencia por lotes, las frases tienen longitudes distintas.
    // El padding rellena con ceros y el attention_mask le dice al Transformer
    // cuáles tokens son reales (1) y cuáles son relleno artificial (0):
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::BatchLongest,
        pad_id: 0,
        pad_token: "<pad>".to_string(),
        ..Default::default()
    }));

    let frases = vec![
        "rust ia",           // 2 tokens -> necesitará 2 pads
        "rust ia ru st",     // 4 tokens -> es la más larga
    ];

    let batch_encodings = tokenizer.encode_batch(frases, true)?;

    println!("Lote de 2 frases alineadas con Padding:");
    for (i, enc) in batch_encodings.iter().enumerate() {
        println!("   • Frase {}:", i);
        println!("     IDs:            {:?}", enc.get_ids());
        println!("     Attention Mask: {:?}", enc.get_attention_mask());
        println!("     Tokens:         {:?}", enc.get_tokens());
    }

    println!("\n--- 5. Persistencia: Guardar y Cargar desde tokenizer.json ---");
    let temp_path = std::env::temp_dir().join("mi_tokenizer.json");
    tokenizer.save(&temp_path, true)?;
    println!("✓ Tokenizador guardado en formato universal: {:?}", temp_path);

    let tokenizer_cargado = Tokenizer::from_file(&temp_path)?;
    println!("✓ Tokenizador recargado desde archivo. Vocabulario: {} tokens.",
        tokenizer_cargado.get_vocab_size(true));

    println!("\n✅ Demostración teórica de Tokenizers completada con éxito.");
    Ok(())
}
