//! 🦀 CLASE 02: TEORÍA Y EJEMPLOS - Tipos de Datos y Conversión Estricta
//! Ejecutar con: cargo run --bin c02_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 02 [TEORÍA]: Tipos de Datos y Conversión en IA");
    println!("========================================================\n");

    // 1. Tipos escalares
    let pixel: u8 = 255;
    let peso: f32 = 0.8542;
    let indice: usize = 0;
    let mascara: bool = true;

    println!("Escalares: pixel (u8)={}, peso (f32)={}, indice (usize)={}, mask={}", pixel, peso, indice, mascara);

    // 2. Casting con 'as'
    let total_loss: f32 = 45.8;
    let total_batches: usize = 10;
    let media = total_loss / (total_batches as f32);
    println!("Casting: Loss acumulada {} / {} batches = {:.4}", total_loss, total_batches, media);

    // 3. Tuplas
    let resultado_modelo: (&str, f32, usize) = ("ResNet-50", 0.942, 50);
    println!("Tupla: Modelo {}, Precisión {:.2}%, Épocas {}", resultado_modelo.0, resultado_modelo.1 * 100.0, resultado_modelo.2);

    let (modelo, acc, epocas) = resultado_modelo;
    println!("Destructuración: {} acc={:.2}% tras {} epocas", modelo, acc * 100.0, epocas);

    // 4. Arrays fijos [T; N]
    let shape_tensor: [usize; 3] = [32, 224, 224];
    let ceros: [f32; 4] = [0.0; 4];
    println!("Array Shape: {:?}", shape_tensor);
    println!("Array Ceros: {:?}", ceros);

    println!("\n✅ Teoría de la Clase 02 completada.");
}
