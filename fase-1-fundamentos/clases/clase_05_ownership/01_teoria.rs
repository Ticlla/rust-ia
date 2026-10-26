//! 🦀 CLASE 05: TEORÍA Y EJEMPLOS - Ownership en Rust para IA
//! Ejecutar con: cargo run --bin c05_teoria

fn main() {
    println!("========================================================");
    println!("📚 CLASE 05 [TEORÍA]: Ownership y Semántica Move para IA");
    println!("========================================================\n");

    // 1. Tipos Copy en el Stack (No pierden propiedad)
    let peso_original: f32 = 0.5;
    let peso_copia = peso_original; // Se copia a nivel de bytes en el Stack
    println!("Stack (Copy): original={}, copia={}", peso_original, peso_copia);

    // 2. Tipos Move en el Heap (Vec, String)
    let tensor_heap = vec![1.0, 2.0, 3.0, 4.0];
    let nuevo_dueno = tensor_heap; // Se transfieren los 24 bytes del encabezado (MOVE)
    // println!("{:?}", tensor_heap); // ❌ ¡Prohibido! tensor_heap ya no es dueño
    println!("Heap (Move): nuevo_dueno posee {} elementos en el Heap", nuevo_dueno.len());

    // 3. Pasar propiedad a funciones
    procesar_dataset(nuevo_dueno); // La función se convierte en dueña y libera la memoria al terminar

    // 4. Clonar cuando realmente necesitamos un duplicado independiente
    let config = vec![128, 256, 512];
    let config_clonada = config.clone(); // Duplica físicamente el buffer en el Heap
    println!("Clone explícito: config={:?}, clon={:?}", config, config_clonada);

    println!("\n✅ Teoría de la Clase 05 completada.");
}

fn procesar_dataset(datos: Vec<f64>) {
    println!("  [procesar_dataset] Consumiendo {} muestras y liberando memoria.", datos.len());
}
