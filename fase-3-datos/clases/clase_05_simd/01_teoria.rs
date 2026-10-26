//! ==============================================================================
//! CLASE 05: Vectorización y SIMD Explícito (AVX2 / AVX-512)
//! ==============================================================================
//!
//! En Inteligencia Artificial, el 90% de las operaciones son sumas y multiplicaciones
//! de vectores (como el producto punto para la atención o la similitud coseno).
//!
//! Por defecto, una CPU ejecuta instrucciones **Escalares**: toma 1 float, lo multiplica
//! por otro float y produce 1 resultado por ciclo de reloj.
//!
//! **SIMD (Single Instruction, Multiple Data)** cambia las reglas del juego:
//! Usando registros especiales de 256 bits (`ymm`), la CPU procesa **8 floats a la vez**
//! en una sola instrucción de hardware.
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Entender la diferencia entre Cómputo Escalar vs Vectorial (SIMD).
//! 2. Conocer los registros AVX2 (256 bits = 8x f32) y AVX-512 (512 bits = 16x f32).
//! 3. Instrucción FMA (Fused Multiply-Add): a * b + c en un solo ciclo de reloj.
//! 4. Benchmark en vivo: Medir la aceleración de velocidad real en tu procesador.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::time::Instant;

fn main() {
    println!("============================================================");
    println!("⚡ CLASE 05: Vectorización y SIMD (AVX2 / Hardware Acceleration)");
    println!("============================================================\n");

    demostrar_concepto_simd();
    ejecutar_benchmark_en_vivo();

    println!("\n✅ Demostración teórica de SIMD completada con éxito.");
}

/// ----------------------------------------------------------------------------
/// 1. ESCALAR VS VECTORIAL (SIMD)
/// ----------------------------------------------------------------------------
fn demostrar_concepto_simd() {
    println!("--- 1. Anatomía de Hardware: Escalar vs SIMD (AVX2) ---");
    println!("🐢 MODO ESCALAR (CPU Tradicional):");
    println!("   Registro: [ a0 ]  x  [ b0 ]  =  [ c0 ]  (1 operación por ciclo)");
    println!("   Para 8 números: La CPU necesita 8 ciclos de reloj.\n");

    println!("🚀 MODO VECTORIAL SIMD (Registros AVX2 de 256 bits):");
    println!("   Registro YMM1: [ a0 │ a1 │ a2 │ a3 │ a4 │ a5 │ a6 │ a7 ] (8 floats)");
    println!("   Registro YMM2: [ b0 │ b1 │ b2 │ b3 │ b4 │ b5 │ b6 │ b7 ] (8 floats)");
    println!("   Instrucción:   MULPS (Multiply Packed Single-Precision)");
    println!("   Resultado:     [ c0 │ c1 │ c2 │ c3 │ c4 │ c5 │ c6 │ c7 ]");
    println!("   👉 ¡Los 8 números se multiplican SIMULTÁNEAMENTE en 1 solo ciclo!\n");

    #[cfg(target_arch = "x86_64")]
    {
        println!("🔍 Capacidades de hardware detectadas en tu procesador:");
        println!("   • Soporta AVX2:    {}", is_x86_feature_detected!("avx2"));
        println!("   • Soporta FMA:     {}", is_x86_feature_detected!("fma"));
        println!("   • Soporta AVX-512: {}", is_x86_feature_detected!("avx512f"));
        println!();
    }
}

/// ----------------------------------------------------------------------------
/// 2. IMPLEMENTACIÓN: PRODUCTO PUNTO ESCALAR (TRADICIONAL)
/// ----------------------------------------------------------------------------
pub fn producto_punto_escalar(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut suma = 0.0f32;
    for i in 0..a.len() {
        suma += a[i] * b[i];
    }
    suma
}

/// ----------------------------------------------------------------------------
/// 3. IMPLEMENTACIÓN: PRODUCTO PUNTO SIMD (AVX2 + FMA)
/// ----------------------------------------------------------------------------
/// Procesa bloques de 8 floats a la vez usando Fused Multiply-Add (_mm256_fmadd_ps).
/// La instrucción hardware FMA hace: acumulador = (a * b) + acumulador en 1 ciclo.
#[cfg(target_arch = "x86_64")]
pub unsafe fn producto_punto_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let trozos = len / 8;
    let resto = len % 8;

    // Inicializamos un registro YMM de 256 bits lleno de ceros: [0.0; 8]
    let mut acc_vector = unsafe { _mm256_setzero_ps() };

    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    // Bucle SIMD principal: 8 floats por iteración
    for _ in 0..trozos {
        unsafe {
            // Cargar 8 floats desde la memoria RAM a los registros de la CPU:
            let va = _mm256_loadu_ps(ptr_a);
            let vb = _mm256_loadu_ps(ptr_b);

            // Fused Multiply-Add (FMA): acc = (va * vb) + acc
            acc_vector = _mm256_fmadd_ps(va, vb, acc_vector);

            ptr_a = ptr_a.add(8);
            ptr_b = ptr_b.add(8);
        }
    }

    // Sumar los 8 valores del acumulador vector hacia un solo float escalar:
    let mut buffer = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buffer.as_mut_ptr(), acc_vector);
    }
    let mut suma_total: f32 = buffer.iter().sum();

    // Procesar los elementos que sobraron (el residuo si len no es múltiplo de 8):
    let inicio_resto = trozos * 8;
    for i in 0..resto {
        suma_total += a[inicio_resto + i] * b[inicio_resto + i];
    }

    suma_total
}

/// ----------------------------------------------------------------------------
/// 4. BENCHMARK EN VIVO: MEDICIÓN DE VELOCIDAD
/// ----------------------------------------------------------------------------
fn ejecutar_benchmark_en_vivo() {
    println!("--- 2. Benchmark en Vivo: Multiplicando 2 Millones de Floats ---");

    // Simulamos dos vectores grandes de embeddings / pesos de IA normalizados:
    let n = 2_000_000;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) / (n as f32)).collect();
    let b: Vec<f32> = (0..n).map(|i| ((n - i) as f32) / (n as f32)).collect();

    // 1. Benchmark Escalar:
    let t0 = Instant::now();
    let res_escalar = producto_punto_escalar(&a, &b);
    let duracion_escalar = t0.elapsed();

    println!("🐢 Escalar (1 float a la vez):");
    println!("   Resultado: {:.4}", res_escalar);
    println!("   Tiempo:    {:.3?}", duracion_escalar);

    // 2. Benchmark SIMD AVX2:
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let t1 = Instant::now();
            let res_simd = unsafe { producto_punto_avx2(&a, &b) };
            let duracion_simd = t1.elapsed();

            println!("\n🚀 SIMD AVX2 + FMA (8 floats a la vez):");
            println!("   Resultado: {:.4}", res_simd);
            println!("   Tiempo:    {:.3?}", duracion_simd);

            // En precisión float32 (24 bits de mantisa), acumular 2 millones de números
            // en 8 carriles paralelos con FMA acumula menor error de redondeo que el bucle
            // escalar secuencial. Una diferencia de ~0.1% es esperada por la no-asociatividad flotante:
            let diff_relativa = ((res_escalar - res_simd) / res_escalar).abs();
            assert!(diff_relativa < 2e-3, "Diferencia relativa: {}", diff_relativa);

            let speedup = duracion_escalar.as_secs_f64() / duracion_simd.as_secs_f64();
            println!("\n⚡ ¡Aceleración de Hardware: {:.2}x!", speedup);
        }
    }
}
