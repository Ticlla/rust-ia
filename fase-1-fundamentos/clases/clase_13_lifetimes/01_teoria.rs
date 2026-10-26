//! 🦀 CLASE 13: TEORÍA Y EJEMPLOS - Lifetimes Básicos ('a) en IA
//! Ejecutar con: cargo run --bin c13_teoria

fn mejor_batch<'a>(a: &'a [f32], b: &'a [f32]) -> &'a [f32] {
    let media_a: f32 = a.iter().sum::<f32>() / a.len() as f32;
    let media_b: f32 = b.iter().sum::<f32>() / b.len() as f32;
    if media_a < media_b { a } else { b }
}

#[allow(dead_code)]
struct TensorView<'a> {
    buffer: &'a [f32],
    filas: usize,
    columnas: usize,
}

impl<'a> TensorView<'a> {
    fn obtener(&self, fila: usize, col: usize) -> f32 {
        self.buffer[fila * self.columnas + col]
    }
}

fn main() {
    println!("========================================================");
    println!("📚 CLASE 13 [TEORÍA]: Lifetimes Básicos ('a) para IA");
    println!("========================================================\n");

    let b1 = [0.85, 0.90];
    let b2 = [0.40, 0.50];
    let mejor = mejor_batch(&b1, &b2);
    println!("Mejor batch seleccionado: {:?}", mejor);

    let datos = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let vista = TensorView { buffer: &datos, filas: 2, columnas: 3 };
    println!("TensorView elemento (1, 2): {}", vista.obtener(1, 2));

    println!("\n✅ Teoría de la Clase 13 completada.");
}
