//! 🦀 CLASE 10: TEORÍA Y EJEMPLOS - Errores (Option y Result) en IA
//! Ejecutar con: cargo run --bin c10_teoria

fn dividir_loss(total: f32, count: usize) -> Result<f32, String> {
    if count == 0 {
        Err("No se puede dividir por cero batches".to_string())
    } else {
        Ok(total / count as f32)
    }
}

fn buscar_muestra(muestras: &[f32], idx: usize) -> Option<f32> {
    if idx < muestras.len() {
        Some(muestras[idx])
    } else {
        None
    }
}

fn main() {
    println!("========================================================");
    println!("📚 CLASE 10 [TEORÍA]: Manejo Robusto de Errores para IA");
    println!("========================================================\n");

    // 1. Result
    match dividir_loss(100.0, 10) {
        Ok(promedio) => println!("Result OK: Media = {:.2}", promedio),
        Err(e) => println!("Result ERR: {}", e),
    }

    match dividir_loss(100.0, 0) {
        Ok(_) => (),
        Err(e) => println!("División por cero capturada: {}", e),
    }

    // 2. Option
    let datos = [0.85, 0.92, 0.78];
    if let Some(valor) = buscar_muestra(&datos, 1) {
        println!("Option SOME: Muestra encontrada = {}", valor);
    }

    println!("\n✅ Teoría de la Clase 10 completada.");
}
