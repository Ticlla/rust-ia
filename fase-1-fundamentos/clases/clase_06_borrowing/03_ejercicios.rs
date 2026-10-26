//! 🦀 CLASE 06: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c06_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 06 [EJERCICIOS PRÁCTICOS]: Borrowing");
    println!("========================================================\n");

    // Reto 1: Inferencia de solo lectura (Cero copias)
    println!("--- Reto 1: Inferencia con Referencia Inmutable ---");
    let tensor = vec![0.8, -0.4, 0.9];
    let suma = sumar_elementos(&tensor);
    assert!((suma - 1.3).abs() < 1e-5);
    println!("Tensor intacto: {:?} | Suma: {:.2}\n", tensor, suma);

    // Reto 2: Normalización in-place con &mut
    println!("--- Reto 2: Modificación in-place con &mut ---");
    let mut pesos = vec![2.0, 4.0, 6.0];
    escalar_vector(&mut pesos, 0.5);
    assert_eq!(pesos, vec![1.0, 2.0, 3.0]);
    println!("Pesos escalados: {:?}\n", pesos);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 06 FUERON SUPERADOS!");
    println!("========================================================");
}

fn sumar_elementos(datos: &Vec<f64>) -> f64 {
    datos.iter().sum()
}

fn escalar_vector(datos: &mut Vec<f64>, factor: f64) {
    for val in datos.iter_mut() {
        *val *= factor;
    }
}
