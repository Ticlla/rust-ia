//! 🦀 CLASE 09: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c09_ejercicios

#[allow(dead_code)]
enum Activacion {
    ReLU,
    Sigmoide,
    Lineal,
}

impl Activacion {
    fn aplicar(&self, x: f32) -> f32 {
        match self {
            Activacion::ReLU => if x > 0.0 { x } else { 0.0 },
            Activacion::Sigmoide => 1.0 / (1.0 + (-x).exp()),
            Activacion::Lineal => x,
        }
    }
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 09 [EJERCICIOS PRÁCTICOS]: Enums");
    println!("========================================================\n");

    let relu = Activacion::ReLU;
    let lineal = Activacion::Lineal;

    assert_eq!(relu.aplicar(-5.0), 0.0);
    assert_eq!(relu.aplicar(3.0), 3.0);
    assert_eq!(lineal.aplicar(7.5), 7.5);

    println!("Activaciones evaluadas correctamente con Enums.\n");
    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 09 FUERON SUPERADOS!");
    println!("========================================================");
}
