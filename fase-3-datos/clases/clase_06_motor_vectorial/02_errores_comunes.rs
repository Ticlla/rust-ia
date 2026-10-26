//! ==============================================================================
//! CLASE 06: Errores Comunes en Motores de Búsqueda Vectorial (RAG)
//! ==============================================================================
//!
//! Al diseñar motores de búsqueda vectorial de alto rendimiento, los cuellos
//! de botella suelen originarse en alocaciones en bucles calientes o escaneos
//! masivos innecesarios.

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones en Búsqueda Vectorial");
    println!("============================================================\n");

    error_1_alocaciones_en_el_bucle_caliente();
    error_2_calcular_normas_repetidamente_en_cada_query();
    error_3_escaneo_exhaustivo_sin_prefiltrado();

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Alocaciones Dinámicas (Vec::clone) en el Bucle de Búsqueda
/// ----------------------------------------------------------------------------
/// Si tienes 100,000 documentos y en cada iteración clonas un `Vec<f32>`,
/// estás pidiéndole memoria al sistema operativo 100,000 veces por consulta.
fn error_1_alocaciones_en_el_bucle_caliente() {
    println!("--- Error 1: Alocar Memoria en el Bucle Caliente de Búsqueda ---");
    let matriz_embeddings = vec![0.1f32; 1000 * 8]; // 1000 docs de dimensión 8

    // ❌ ANTIPATRÓN: Crear un Vec nuevo por cada candidato
    // for i in 0..1000 {
    //     let doc_vec: Vec<f32> = matriz_embeddings[i*8..(i+1)*8].to_vec(); // ALOCACIÓN LENTA
    //     ...
    // }

    //  PATRÓN CORRECTO (Cero-Copia con Slices):
    for i in 0..1000 {
        let _doc_slice: &[f32] = &matriz_embeddings[i * 8..(i + 1) * 8]; // Puntero directo (0 ns)
    }

    println!("✓ Slices usados directamente: Cero alocaciones en el heap durante el ranking.\n");
}

/// ----------------------------------------------------------------------------
/// ERROR 2: Calcular Normas y Raíces Cuadradas en Tiempo de Consulta
/// ----------------------------------------------------------------------------
/// Similitud = (a · b) / (||a|| * ||b||)
/// Si calculas ||b|| (la norma del documento) en cada consulta:
/// Estás calculando raíces cuadradas millones de veces para vectores estáticos.
fn error_2_calcular_normas_repetidamente_en_cada_query() {
    println!("--- Error 2: Calcular Normas en Cada Consulta ---");
    println!("❌ Lento: Calcular `dot / (norm_a * norm_b)` en cada uno de los 50,000 docs.");
    println!("  Rápido: Pre-normalizar los embeddings en el momento de la indexación (||doc|| = 1.0).");
    println!("   Así, la similitud coseno es simplemente el Producto Punto: `dot(query, doc)`.\n");
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Búsqueda Exhaustiva sin Pre-filtrado (Brute Force O(N))
/// ----------------------------------------------------------------------------
/// Si un usuario busca documentos en español del año 2024, calcular similitud
/// contra 2 millones de documentos en inglés de 2020 es un desperdicio del 90%
/// de los ciclos de CPU.
fn error_3_escaneo_exhaustivo_sin_prefiltrado() {
    println!("--- Error 3: Búsqueda Exhaustiva sin Pre-filtrado de Metadatos ---");
    println!("💡 La Arquitectura Híbrida (Polars + SIMD):");
    println!("   1. Polars (Apache Arrow) filtra millones de metadatos en microsegundos.");
    println!("   2. SIMD solo evalúa los candidatos que cumplieron las condiciones.");
    println!("   Esto reduce el espacio de búsqueda de 1,000,000 a solo 500 candidatos.");
}
