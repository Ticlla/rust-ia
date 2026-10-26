//! 🦀 CLASE 13: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c13_ejercicios

fn seleccionar_tensor<'a>(t1: &'a [f32], t2: &'a [f32], usar_primero: bool) -> &'a [f32] {
    if usar_primero { t1 } else { t2 }
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 13 [EJERCICIOS PRÁCTICOS]: Lifetimes");
    println!("========================================================\n");

    let a = [1.0, 2.0];
    let b = [3.0, 4.0];
    let elegido = seleccionar_tensor(&a, &b, false);

    assert_eq!(elegido, &[3.0, 4.0]);
    println!("Tensor seleccionado correctamente con 'a: {:?}\n", elegido);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 13 FUERON SUPERADOS!");
    println!("========================================================");
}
