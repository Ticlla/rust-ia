//! 🦀 CLASE 07: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c07_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 07 [EJERCICIOS PRÁCTICOS]: Slices");
    println!("========================================================\n");

    // Reto 1: Norma L1 sobre slice
    println!("--- Reto 1: Norma L1 de un Slice ---");
    let vector = [1.5, -2.5, 3.0];
    let norma = norma_l1(&vector);
    assert!((norma - 7.0).abs() < 1e-5);
    println!("Norma L1: {:.2}\n", norma);

    // Reto 2: Ventana deslizante de 2 elementos
    println!("--- Reto 2: Ventana Deslizante (Sliding Window) ---");
    let serie_temporal = [1.0, 2.0, 4.0, 7.0];
    let mut diferencias = Vec::new();
    for ventana in serie_temporal.windows(2) {
        diferencias.push(ventana[1] - ventana[0]);
    }
    assert_eq!(diferencias, vec![1.0, 2.0, 3.0]);
    println!("Diferencias de ventana: {:?}\n", diferencias);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 07 FUERON SUPERADOS!");
    println!("========================================================");
}

fn norma_l1(slice: &[f64]) -> f64 {
    slice.iter().map(|x| x.abs()).sum()
}
