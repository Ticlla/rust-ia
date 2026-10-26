//! ==============================================================================
//! CLASE 05: Ejercicios Prácticos de Vectorización y SIMD
//! ==============================================================================
//!
//! En este laboratorio implementarás las operaciones más aceleradas de IA:
//! 1. Producto Punto Robusto con AVX2 + FMA (con soporte para cualquier longitud).
//! 2. Similitud Coseno Acelerada por Hardware (para Vector Search / RAG).
//! 3. Suma de Vectores Element-wise con SIMD (para conexiones residuales).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 05: Aceleración por Hardware (SIMD)");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c05_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Producto Punto AVX2 Robusto (Longitud Arbitraria)
/// ----------------------------------------------------------------------------
/// Calcula el producto punto usando registros YMM de 256 bits y FMA si está
/// disponible en hardware, rematando con bucle escalar para el residuo.
pub fn producto_punto_simd(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Longitudes incompatibles");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { producto_punto_avx2_impl(a, b) };
        }
    }

    // Fallback escalar si la CPU no tuviera AVX2:
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(target_arch = "x86_64")]
unsafe fn producto_punto_avx2_impl(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let trozos = len / 8;
    let resto = len % 8;

    let mut acc = unsafe { _mm256_setzero_ps() };
    let mut ptr_a = a.as_ptr();
    let mut ptr_b = b.as_ptr();

    for _ in 0..trozos {
        unsafe {
            let va = _mm256_loadu_ps(ptr_a);
            let vb = _mm256_loadu_ps(ptr_b);
            acc = _mm256_fmadd_ps(va, vb, acc);

            ptr_a = ptr_a.add(8);
            ptr_b = ptr_b.add(8);
        }
    }

    let mut buf = [0.0f32; 8];
    unsafe {
        _mm256_storeu_ps(buf.as_mut_ptr(), acc);
    }
    let mut suma: f32 = buf.iter().sum();

    // Residuo
    let inicio_resto = trozos * 8;
    for i in 0..resto {
        suma += a[inicio_resto + i] * b[inicio_resto + i];
    }

    suma
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Similitud Coseno Acelerada por Hardware (SIMD)
/// ----------------------------------------------------------------------------
/// Similitud = (a · b) / (||a|| * ||b||)
pub fn similitud_coseno_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = producto_punto_simd(a, b);
    let norma_a = producto_punto_simd(a, a).sqrt();
    let norma_b = producto_punto_simd(b, b).sqrt();

    let denominador = norma_a * norma_b;
    if denominador < 1e-12 {
        0.0
    } else {
        dot / denominador
    }
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 3: Suma Vectorial Element-wise (Conexión Residual en Transformers)
/// ----------------------------------------------------------------------------
/// Calcula: c[i] = a[i] + b[i]
pub fn suma_vectorial_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let trozos = len / 8;
    let resto = len % 8;

    let mut resultado = vec![0.0f32; len];

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        let mut ptr_a = a.as_ptr();
        let mut ptr_b = b.as_ptr();
        let mut ptr_res = resultado.as_mut_ptr();

        for _ in 0..trozos {
            unsafe {
                let va = _mm256_loadu_ps(ptr_a);
                let vb = _mm256_loadu_ps(ptr_b);
                let vsum = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(ptr_res, vsum);

                ptr_a = ptr_a.add(8);
                ptr_b = ptr_b.add(8);
                ptr_res = ptr_res.add(8);
            }
        }
    } else {
        for i in 0..(trozos * 8) {
            resultado[i] = a[i] + b[i];
        }
    }

    let inicio_resto = trozos * 8;
    for i in 0..resto {
        resultado[inicio_resto + i] = a[inicio_resto + i] + b[inicio_resto + i];
    }

    resultado
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_producto_punto_simd_longitud_impar() {
        // 11 elementos (1 trozo de 8 + 3 de residuo)
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        let b = vec![2.0f32; 11];

        // Esperado: 2 * (1 + 2 + ... + 11) = 2 * 66 = 132.0
        let res = producto_punto_simd(&a, &b);
        assert!((res - 132.0).abs() < 1e-5);
    }

    #[test]
    fn test_similitud_coseno_simd() {
        let v1 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v2 = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let v3 = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Vectores idénticos -> Similitud 1.0
        assert!((similitud_coseno_simd(&v1, &v2) - 1.0).abs() < 1e-5);

        // Vectores ortogonales -> Similitud 0.0
        assert!((similitud_coseno_simd(&v1, &v3) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_suma_vectorial_simd() {
        let a = vec![1.0f32; 10];
        let b = vec![2.5f32; 10];
        let c = suma_vectorial_simd(&a, &b);

        assert_eq!(c.len(), 10);
        for &val in &c {
            assert!((val - 3.5).abs() < 1e-5);
        }
    }
}
