//! 🦀 CLASE 09: LABORATORIO DE ERRORES COMUNES - Enums
//! Ejecutar con: cargo run --bin c09_errores

#[allow(dead_code)]
enum Optimizador {
    SGD,
    Adam,
    RMSprop,
}

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 09 [LABORATORIO DE ERRORES]: Enums");
    println!("========================================================\n");

    println!("--- Error 1: error[E0004] - Non-exhaustive patterns en match ---");
    println!("❌ Si omites una variante en match (ej. olvidas RMSprop), Rust rehúsa compilar.");
    println!("✅ Solución: Cubrir todas las variantes o usar el comodín '_':\n");

    let opt = Optimizador::RMSprop;
    let desc = match opt {
        Optimizador::SGD => "Descenso de gradiente estocástico",
        Optimizador::Adam => "Adam adaptativo",
        Optimizador::RMSprop => "RMSprop adaptativo",
    };
    println!("   Evaluación exhaustiva segura: {}\n", desc);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
