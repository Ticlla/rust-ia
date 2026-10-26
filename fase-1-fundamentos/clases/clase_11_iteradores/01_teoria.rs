//! 🦀 CLASE 11: TEORÍA Y EJEMPLOS - Iteradores y Closures en IA
//! Ejecutar con: cargo run --bin c11_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 11 [TEORÍA]: Iteradores Funcionales para IA");
    println!("========================================================\n");

    let a = [1.0, 2.0, 3.0];
    let b = [4.0, 5.0, 6.0];

    // Producto punto vectorizado con iteradores (Zero-Cost):
    let producto_punto: f32 = a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum();

    println!("Vector A: {:?}", a);
    println!("Vector B: {:?}", b);
    println!("A · B (Producto Punto): {}", producto_punto);

    // Filtrar pérdidas anómalas
    let perdidas = vec![0.5, 0.4, 999.0, 0.3];
    let perdidas_validas: Vec<f32> = perdidas.into_iter()
        .filter(|&loss| loss < 100.0)
        .collect();
    println!("Pérdidas filtradas: {:?}", perdidas_validas);

    println!("\n✅ Teoría de la Clase 11 completada.");
}
