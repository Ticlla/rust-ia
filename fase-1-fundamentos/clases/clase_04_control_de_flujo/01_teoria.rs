//! 🦀 CLASE 04: TEORÍA Y EJEMPLOS - Control de Flujo para IA
//! Ejecutar con: cargo run --bin c04_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 04 [TEORÍA]: Control de Flujo en IA");
    println!("========================================================\n");

    // 1. loop con retorno de valor (Búsqueda de Hiperparámetro)
    let mut lr = 0.001;
    let mejor_lr = loop {
        lr *= 2.0;
        if lr > 0.05 {
            break lr; // Devuelve este valor fuera del loop
        }
    };
    println!("Mejor LR encontrado con 'loop': {}", mejor_lr);

    // 2. while para convergencia de entrenamiento
    let mut loss = 2.0;
    let mut pasos = 0;
    while loss > 0.2 {
        loss *= 0.7;
        pasos += 1;
    }
    println!("Convergencia alcanzada tras {} pasos (Loss: {:.4})", pasos, loss);

    // 3. for sobre rangos y batches
    let batches = [0.45, 0.38, 0.29, 0.15];
    for (idx, &b_loss) in batches.iter().enumerate() {
        println!("  Batch {}: loss = {:.4}", idx, b_loss);
    }

    println!("\n✅ Teoría de la Clase 04 completada.");
}
