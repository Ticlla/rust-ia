//! 🦀 CLASE 08: TEORÍA Y EJEMPLOS - Structs en Rust para IA
//! Ejecutar con: cargo run --bin c08_teoria

struct Neurona {
    peso: f32,
    sesgo: f32,
}

impl Neurona {
    fn new(peso: f32, sesgo: f32) -> Self {
        Self { peso, sesgo }
    }

    fn forward(&self, x: f32) -> f32 {
        x * self.peso + self.sesgo
    }

    fn actualizar(&mut self, lr: f32, grad_w: f32, grad_b: f32) {
        self.peso -= lr * grad_w;
        self.sesgo -= lr * grad_b;
    }
}

fn main() {
    println!("========================================================");
    println!("📚 CLASE 08 [TEORÍA]: Structs y Métodos para IA");
    println!("========================================================\n");

    let mut n = Neurona::new(0.5, 1.0);
    println!("Neurona inicial: peso={}, sesgo={}", n.peso, n.sesgo);
    println!("Forward(2.0): {}", n.forward(2.0));

    n.actualizar(0.01, 2.0, 1.0);
    println!("Tras optimizar: peso={:.4}, sesgo={:.4}", n.peso, n.sesgo);

    println!("\n✅ Teoría de la Clase 08 completada.");
}
