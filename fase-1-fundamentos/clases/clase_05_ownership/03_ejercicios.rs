//! 🦀 CLASE 05: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c05_ejercicios

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 05 [EJERCICIOS PRÁCTICOS]: Ownership");
    println!("========================================================\n");

    // Reto 1: Función que consume y devuelve la propiedad transformada
    println!("--- Reto 1: Transformador de Dataset con Move ---");
    let buffer = vec![1.0, 2.0, 3.0];
    let buffer_transformado = duplicar_buffer(buffer);
    // 'buffer' fue consumido; 'buffer_transformado' es el nuevo dueño:
    assert_eq!(buffer_transformado, vec![2.0, 4.0, 6.0]);
    println!("Buffer transformado: {:?}\n", buffer_transformado);

    // Reto 2: Copia en Stack vs Clon en Heap
    println!("--- Reto 2: Verificación de tipos Copy vs Clone ---");
    let a = [10, 20, 30]; // Array en Stack (Copy)
    let b = a;
    assert_eq!(a, b); // 'a' sigue vivo porque es un array en el Stack
    println!("Array en stack copiado sin invalidar original: {:?}\n", a);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 05 FUERON SUPERADOS!");
    println!("========================================================");
}

fn duplicar_buffer(mut datos: Vec<f64>) -> Vec<f64> {
    for val in &mut datos {
        *val *= 2.0;
    }
    datos
}
