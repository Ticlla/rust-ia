fn main() {
    println!("========================================================");
    println!("⚙️ FASE 2: SISTEMAS Y CONCURRENCIA PARA INGENIEROS EN IA");
    println!("========================================================\n");
    println!("Cada clase cuenta con su propia suite modular de aprendizaje:\n");
    println!("  • [Teoría]   : cargo run --bin cXX_teoria");
    println!("  • [Errores]  : cargo run --bin cXX_errores");
    println!("  • [Retos]    : cargo run --bin cXX_ejercicios\n");
    println!("----------------------------------------------------------------------------------");
    println!("  ID   TEMA                                TEORÍA        ERRORES       RETOS");
    println!("----------------------------------------------------------------------------------");

    let catalogo = [
        ("01", "Hilos Nativos (std::thread) y move", "c01_teoria", "c01_errores", "c01_ejercicios"),
        ("02", "Canales (mpsc): DataLoader Paralelo", "c02_teoria", "c02_errores", "c02_ejercicios"),
        ("03", "Memoria Compartida: Arc<Mutex<T>>", "c03_teoria", "c03_errores", "c03_ejercicios"),
        ("04", "Lecturas Concurrentes: RwLock<T>", "c04_teoria", "c04_errores", "c04_ejercicios"),
        ("05", "Paralelismo de Datos con Rayon", "c05_teoria", "c05_errores", "c05_ejercicios"),
        ("06", "Asincronía Básica (async / await)", "c06_teoria", "c06_errores", "c06_ejercicios"),
        ("07", "Runtime Asíncrono con Tokio", "c07_teoria", "c07_errores", "c07_ejercicios"),
        ("08", "Servidores de Inferencia con Axum", "c08_teoria", "c08_errores", "c08_ejercicios"),
        ("09", "Punteros Crudos y unsafe Rust", "c09_teoria", "c09_errores", "c09_ejercicios"),
        ("10", "PROYECTO: Pipeline y Servidor IA", "c10_proyecto", "c10_errores", "c10_ejercicios"),
    ];

    for (id, tema, teoria, errores, retos) in &catalogo {
        println!("  {:>2}   {:<35} {:<13} {:<13} {:<13}", id, tema, teoria, errores, retos);
    }

    println!("----------------------------------------------------------------------------------");
    println!("\n💡 Comandos para comenzar la Clase 01:");
    println!("  cargo run --bin c01_teoria        # Aprender hilos y paralelismo");
    println!("  cargo run --bin c01_errores       # Ver errores comunes del compilador");
    println!("  cargo run --bin c01_ejercicios     # Reto: Producto punto en 2 hilos");
    println!("  cargo test --bin c01_ejercicios    # Correr tests unitarios\n");
}
