//! 🦀 CLASE 09: TEORÍA Y EJEMPLOS - Enums en Rust para IA
//! Ejecutar con: cargo run --bin c09_teoria

#[allow(dead_code)]
enum EstadoModelo {
    Inicializando,
    Entrenando { epoca: usize, loss: f32 },
    Convergido(f32),
    Error(String),
}

fn describir_estado(estado: &EstadoModelo) {
    match estado {
        EstadoModelo::Inicializando => println!("El modelo se está configurando en GPU."),
        EstadoModelo::Entrenando { epoca, loss } => {
            println!("Entrenando: Época {}, Pérdida: {:.4}", epoca, loss);
        }
        EstadoModelo::Convergido(loss_final) => {
            println!("✅ Convergencia exitosa con Loss final: {:.6}", loss_final);
        }
        EstadoModelo::Error(err) => println!("❌ Fallo: {}", err),
    }
}

fn main() {
    println!("========================================================");
    println!("📚 CLASE 09 [TEORÍA]: Enums y Pattern Matching para IA");
    println!("========================================================\n");

    let e1 = EstadoModelo::Inicializando;
    let e2 = EstadoModelo::Entrenando { epoca: 10, loss: 0.125 };
    let e3 = EstadoModelo::Convergido(0.002);

    describir_estado(&e1);
    describir_estado(&e2);
    describir_estado(&e3);

    println!("\n✅ Teoría de la Clase 09 completada.");
}
