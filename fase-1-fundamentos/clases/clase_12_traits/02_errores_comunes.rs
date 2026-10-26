//! 🦀 CLASE 12: LABORATORIO DE ERRORES COMUNES - Traits
//! Ejecutar con: cargo run --bin c12_errores

trait Evaluador {
    fn score(&self) -> f32;
}

struct MiModelo;

impl Evaluador for MiModelo {
    fn score(&self) -> f32 { 0.95 }
}

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 12 [LABORATORIO DE ERRORES]: Traits");
    println!("========================================================\n");

    println!("--- Error 1: error[E0046] - Método obligatorio no implementado ---");
    println!("❌ Si un trait define un método sin cuerpo por defecto, estás obligado a implementarlo.");
    println!("✅ Solución: Implementar todas las funciones requeridas por el contrato:");
    let m = MiModelo;
    println!("   Score implementado con éxito: {}\n", m.score());

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
