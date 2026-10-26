//! 🦀 CLASE 02: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c02_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 02 [EJERCICIOS PRÁCTICOS]: Tipos de Datos");
    println!("========================================================\n");

    // Reto 1: Normalización de Píxel de Imagen
    println!("--- Reto 1: Normalización RGB a [0.0, 1.0] ---");
    let pixel_crudo: u8 = 128;
    let pixel_norm: f32 = pixel_crudo as f32 / 255.0;
    println!("Píxel u8: {} -> f32 normalizado: {:.4}", pixel_crudo, pixel_norm);
    assert!((pixel_norm - (128.0 / 255.0)).abs() < 1e-5);
    println!("✅ Reto 1 completado con éxito!\n");

    // Reto 2: Desestructuración de Tupla de Inferencia
    println!("--- Reto 2: Extracción de métricas de un modelo ---");
    let checkpoint: (&str, f32, f32) = ("Transformer-V2", 0.954, 0.046);
    let (nombre, accuracy, val_loss) = checkpoint;
    println!("Modelo: {} | Accuracy: {:.2}% | Loss: {:.4}", nombre, accuracy * 100.0, val_loss);
    assert_eq!(nombre, "Transformer-V2");
    println!("✅ Reto 2 completado con éxito!\n");

    // Reto 3: Array 2D para Tensor 2x2
    println!("--- Reto 3: Matriz de Pesos 2x2 ---");
    let matriz_pesos: [[f32; 2]; 2] = [
        [0.5, -0.2],
        [0.8,  0.1],
    ];
    let suma_diagonal = matriz_pesos[0][0] + matriz_pesos[1][1];
    println!("Traza de la matriz (diag sum): {:.2}", suma_diagonal);
    assert!((suma_diagonal - 0.6).abs() < 1e-5);
    println!("✅ Reto 3 completado con éxito!\n");

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 02 FUERON SUPERADOS!");
    println!("========================================================");
}
