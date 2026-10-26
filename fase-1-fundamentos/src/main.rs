fn main() {
    println!("========================================================");
    println!("🦀 FASE 1: FUNDAMENTOS DE RUST PARA INGENIEROS EN IA");
    println!("========================================================\n");
    println!("Cada clase cuenta con su propia suite modular de aprendizaje:\n");
    println!("  • [Teoría]   : cargo run --bin cXX_teoria");
    println!("  • [Errores]  : cargo run --bin cXX_errores");
    println!("  • [Retos]    : cargo run --bin cXX_ejercicios\n");
    println!("----------------------------------------------------------------------------------");
    println!("  ID   TEMA                                TEORÍA        ERRORES       RETOS");
    println!("----------------------------------------------------------------------------------");

    let catalogo = [
        ("01", "Variables, Mutabilidad y Shadowing", "c01_teoria", "c01_errores", "c01_ejercicios"),
        ("02", "Tipos Escalares, Compuestos y Cast", "c02_teoria", "c02_errores", "c02_ejercicios"),
        ("03", "Funciones, Expresiones y Sentencias", "c03_teoria", "c03_errores", "c03_ejercicios"),
        ("04", "Control de Flujo (loop, while, for)", "c04_teoria", "c04_errores", "c04_ejercicios"),
        ("05", "Ownership: Stack vs Heap y Move", "c05_teoria", "c05_errores", "c05_ejercicios"),
        ("06", "Borrowing y Referencias (& vs &mut)", "c06_teoria", "c06_errores", "c06_ejercicios"),
        ("07", "Slices y Buffers de Memoria Continua", "c07_teoria", "c07_errores", "c07_ejercicios"),
        ("08", "Structs y Métodos Asociados (impl)", "c08_teoria", "c08_errores", "c08_ejercicios"),
        ("09", "Enums y Pattern Matching (match)", "c09_teoria", "c09_errores", "c09_ejercicios"),
        ("10", "Manejo Robusto de Errores (Option/Result)", "c10_teoria", "c10_errores", "c10_ejercicios"),
        ("11", "Iteradores y Closures Funcionales", "c11_teoria", "c11_errores", "c11_ejercicios"),
        ("12", "Traits y Abstracciones en IA", "c12_teoria", "c12_errores", "c12_ejercicios"),
        ("13", "Lifetimes Básicos ('a) y Zero-Copy", "c13_teoria", "c13_errores", "c13_ejercicios"),
        ("14", "PROYECTO FINAL: Regresión Lineal", "c14_proyecto", "c14_errores", "c14_ejercicios"),
    ];

    for (id, tema, teoria, errores, retos) in &catalogo {
        println!("  {:>2}   {:<35} {:<13} {:<13} {:<13}", id, tema, teoria, errores, retos);
    }

    println!("----------------------------------------------------------------------------------");
    println!("\n💡 Ejemplos de uso rápido:");
    println!("  cargo run --bin c01_teoria        # Aprender conceptos");
    println!("  cargo run --bin c05_errores       # Probar el laboratorio de Ownership");
    println!("  cargo run --bin c14_interactivo   # 🎮 PROYECTO INTERACTIVO: Entrenar y predecir en vivo");
    println!("  cargo run --bin c14_proyecto      # Entrenar la Regresión Lineal formal desde cero\n");
}
