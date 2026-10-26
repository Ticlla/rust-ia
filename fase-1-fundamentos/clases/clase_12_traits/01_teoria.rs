//! 🦀 CLASE 12: TEORÍA Y EJEMPLOS - Traits en IA
//! Ejecutar con: cargo run --bin c12_teoria

trait ModeloPredictivo {
    fn predecir(&self, x: f32) -> f32;
    fn nombre(&self) -> &str;
}

struct ModeloLineal {
    peso: f32,
    sesgo: f32,
}

impl ModeloPredictivo for ModeloLineal {
    fn predecir(&self, x: f32) -> f32 {
        x * self.peso + self.sesgo
    }
    fn nombre(&self) -> &str {
        "Regresión Lineal Simple"
    }
}

fn probar_modelo(m: &impl ModeloPredictivo, x: f32) {
    println!("  [{}] Inferencia para x={}: y={:.2}", m.nombre(), x, m.predecir(x));
}

fn main() {
    println!("========================================================");
    println!("📚 CLASE 12 [TEORÍA]: Traits y Polimorfismo para IA");
    println!("========================================================\n");

    let lineal = ModeloLineal { peso: 2.0, sesgo: 1.0 };
    probar_modelo(&lineal, 3.0);

    println!("\n✅ Teoría de la Clase 12 completada.");
}
