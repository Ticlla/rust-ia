//! 🦀 CLASE 03: TEORÍA Y EJEMPLOS - Funciones en Rust para IA
//! Ejecutar con: cargo run --bin c03_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 03 [TEORÍA]: Funciones y Expresiones para IA");
    println!("========================================================\n");

    let x = 2.0;
    let w = 0.5;
    let b = 1.0;
    let y = calcular_neurona(x, w, b);
    println!("Neurona: {} * {} + {} = {}", x, w, b, y);

    // Bloque como expresión
    let penalty = {
        let alpha = 0.01;
        let l2 = 0.005;
        alpha * l2
    };
    println!("L2 Penalty evaluado en bloque: {}", penalty);

    // Funciones de activación
    println!("ReLU(-3.5) = {}", relu(-3.5));
    println!("ReLU( 4.2) = {}", relu(4.2));
    println!("Leaky ReLU(-2.0) = {}", leaky_relu(-2.0, 0.01));

    println!("\n✅ Teoría de la Clase 03 completada.");
}

fn calcular_neurona(x: f32, w: f32, b: f32) -> f32 {
    x * w + b
}

fn relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn leaky_relu(x: f32, alpha: f32) -> f32 {
    if x > 0.0 { x } else { alpha * x }
}
