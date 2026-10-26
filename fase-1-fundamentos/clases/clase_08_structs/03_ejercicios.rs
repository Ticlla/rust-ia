//! 🦀 CLASE 08: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c08_ejercicios

struct OptimizadorSGD {
    learning_rate: f32,
    paso: usize,
}

impl OptimizadorSGD {
    fn new(lr: f32) -> Self {
        Self { learning_rate: lr, paso: 0 }
    }

    fn step(&mut self, parametro: &mut f32, gradiente: f32) {
        *parametro -= self.learning_rate * gradiente;
        self.paso += 1;
    }
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 08 [EJERCICIOS PRÁCTICOS]: Structs");
    println!("========================================================\n");

    let mut opt = OptimizadorSGD::new(0.1);
    let mut peso = 5.0;

    opt.step(&mut peso, 2.0); // 5.0 - (0.1 * 2.0) = 4.8
    assert!((peso - 4.8).abs() < 1e-5);
    assert_eq!(opt.paso, 1);

    println!("Peso optimizado: {:.2} tras paso {}\n", peso, opt.paso);
    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 08 FUERON SUPERADOS!");
    println!("========================================================");
}
