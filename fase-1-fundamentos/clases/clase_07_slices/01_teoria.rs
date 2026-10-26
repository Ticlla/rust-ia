//! 🦀 CLASE 07: TEORÍA Y EJEMPLOS - Slices en Rust para IA
//! Ejecutar con: cargo run --bin c07_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 07 [TEORÍA]: Slices y Buffers Continuos para IA");
    println!("========================================================\n");

    let dataset = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];

    // Slices por rangos [inicio..fin_exclusivo]
    let batch_1: &[f64] = &dataset[0..3]; // [10.0, 20.0, 30.0]
    let batch_2: &[f64] = &dataset[3..6]; // [40.0, 50.0, 60.0]

    println!("Dataset completo: {:?}", dataset);
    println!("Batch 1 (0..3)  : {:?}", batch_1);
    println!("Batch 2 (3..6)  : {:?}", batch_2);
    println!("Media Batch 1   : {:.2}", calcular_media(batch_1));
    println!("Media Batch 2   : {:.2}", calcular_media(batch_2));

    println!("\n✅ Teoría de la Clase 07 completada.");
}

fn calcular_media(slice: &[f64]) -> f64 {
    let suma: f64 = slice.iter().sum();
    suma / slice.len() as f64
}
