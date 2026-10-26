//! ==============================================================================
//! CLASE 05: Errores Comunes y Antipatrones de Vectorización y SIMD
//! ==============================================================================
//!
//! Al trabajar con aceleración vectorial directa por hardware, los errores
//! pueden provocar fallos de segmentación (SegFaults) o cálculos incompletos.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Antipatrones de Vectorización SIMD");
    println!("============================================================\n");

    error_1_memoria_desalineada_load_vs_loadu();
    error_2_el_olvido_del_residuo();
    error_3_compilar_sin_target_cpu_native();

    println!("\n✅ Todos los antipatrones analizados y resueltos correctamente.");
}

/// ----------------------------------------------------------------------------
/// ERROR 1: Desalineación de Memoria (_mm256_load_ps vs _mm256_loadu_ps)
/// ----------------------------------------------------------------------------
/// La instrucción `_mm256_load_ps` asume que la dirección de memoria es
/// exactamente múltiplo de 32 bytes. Si no lo está, el procesador dispara
/// un `General Protection Fault` (SegFault) y cuelga el programa.
fn error_1_memoria_desalineada_load_vs_loadu() {
    println!("--- Error 1: Alineación de Memoria (_mm256_loadu_ps) ---");
    println!("❌ Peligroso: `_mm256_load_ps(ptr)`");
    println!("   Crashea si el vector no empezó exactamente en una dirección múltiplo de 32.");
    println!("  Seguro y Rápido: `_mm256_loadu_ps(ptr)`");
    println!("   La 'u' significa Unaligned. En procesadores modernos (Intel 10th Gen+),");
    println!("   no tiene penalización de velocidad cuando los datos están en caché L1.\n");
}

/// ----------------------------------------------------------------------------
/// ERROR 2: El Olvido del Residuo (Remainder)
/// ----------------------------------------------------------------------------
/// Si un embedding tiene dimensión 770 y procesamos en trozos de 8:
/// 770 / 8 = 96 trozos (procesa 768 números).
/// ¡Quedan 2 números huérfanos al final!
/// Si no procesas el residuo, tu cálculo de similitud coseno dará resultados falsos.
fn error_2_el_olvido_del_residuo() {
    println!("--- Error 2: El Manejo Obligatorio del Residuo ---");

    let a = vec![1.0f32; 10]; // 10 elementos (no es múltiplo de 8)
    let b = vec![2.0f32; 10];

    let mut dot = 0.0f32;
    let len = a.len();
    let trozos = len / 8; // 1 trozo de 8
    let resto = len % 8;  // 2 elementos sobrantes

    #[cfg(target_arch = "x86_64")]
    unsafe {
        let va = _mm256_loadu_ps(a.as_ptr());
        let vb = _mm256_loadu_ps(b.as_ptr());
        let mult = _mm256_mul_ps(va, vb);

        let mut buf = [0.0f32; 8];
        _mm256_storeu_ps(buf.as_mut_ptr(), mult);
        dot += buf.iter().sum::<f32>(); // Suma los primeros 8 elementos = 16.0
    }

    // 💡 REMATE ESCALAR OBLIGATORIO:
    let inicio_resto = trozos * 8;
    for i in 0..resto {
        dot += a[inicio_resto + i] * b[inicio_resto + i]; // Suma los 2 restantes = 4.0
    }

    assert_eq!(dot, 20.0);
    println!("✓ Producto punto de 10 elementos calculado correctamente: {}", dot);
    println!("👉 Regla: Siempre agrega el bucle escalar de remate para el `len % 8`.\n");
}

/// ----------------------------------------------------------------------------
/// ERROR 3: Compilar sin `target-cpu=native`
/// ----------------------------------------------------------------------------
/// Por defecto, Rust compila para un x86 genérico compatible con CPUs viejas de 2003.
/// Para liberar el 100% de la potencia de tu CPU Intel Core i5 con AVX2 y AVX-512:
fn error_3_compilar_sin_target_cpu_native() {
    println!("--- Error 3: Desbloquear el 100% de tu CPU con target-cpu ---");
    println!("💡 Comando para compilar liberando AVX2 y AVX-512 en tu máquina:");
    println!("   RUSTFLAGS=\"-C target-cpu=native\" cargo build --release");
    println!("   Esto hace que el propio compilador auto-vectorice bucles normales");
    println!("   incluso sin escribir intrínsecos manuales.");
}
