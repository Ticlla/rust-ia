//! 🦀 CLASE 12: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c12_ejercicios

trait Normalizador {
    fn normalizar(&self, x: f32) -> f32;
}

struct MinMaxNormalizador {
    min: f32,
    max: f32,
}

impl Normalizador for MinMaxNormalizador {
    fn normalizar(&self, x: f32) -> f32 {
        (x - self.min) / (self.max - self.min)
    }
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 12 [EJERCICIOS PRÁCTICOS]: Traits");
    println!("========================================================\n");

    let norm = MinMaxNormalizador { min: 0.0, max: 100.0 };
    let res = norm.normalizar(50.0);
    assert!((res - 0.5).abs() < 1e-5);
    println!("Normalización MinMax (0..100) para 50.0 = {:.2}\n", res);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 12 FUERON SUPERADOS!");
    println!("========================================================");
}
