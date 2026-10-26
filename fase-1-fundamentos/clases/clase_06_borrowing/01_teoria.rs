//! 🦀 CLASE 06: TEORÍA Y EJEMPLOS - Borrowing (& y &mut) en IA
//! Ejecutar con: cargo run --bin c06_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 06 [TEORÍA]: Borrowing y Referencias para IA");
    println!("========================================================\n");

    // 1. Múltiples préstamos inmutables concurrentes (&)
    let dataset = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let lector1 = &dataset;
    let lector2 = &dataset;
    println!("Lectura simultánea: l1 len={}, l2 len={}", lector1.len(), lector2.len());

    // 2. Préstamo mutable exclusivo (&mut)
    let mut modelo_pesos = vec![0.1, 0.2, 0.3];
    println!("Pesos antes de optimizar: {:?}", modelo_pesos);
    aplicar_gradiente(&mut modelo_pesos, 0.05);
    println!("Pesos tras optimizar: {:?}", modelo_pesos);

    println!("\n✅ Teoría de la Clase 06 completada.");
}

fn aplicar_gradiente(pesos: &mut Vec<f64>, gradiente: f64) {
    for p in pesos.iter_mut() {
        *p -= gradiente;
    }
}
