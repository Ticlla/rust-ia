//! ============================================================================
//! 🎯 FASE 2 - CLASE 09: Retos Prácticos con `unsafe` y Tensores en IA
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c09_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c09_ejercicios

use std::alloc::{alloc, dealloc, Layout};

/// Estructura que encapsula memoria contigua para tensores de IA.
/// Provee operaciones vectoriales ultra rápidas con aritmética de punteros sin bounds checking.
pub struct TensorContinuo {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

impl TensorContinuo {
    /// Crea un nuevo tensor continuo inicializado con ceros.
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "El tamaño debe ser mayor que cero");
        let layout = Layout::array::<f32>(len).expect("Layout de memoria inválido");
        let ptr = unsafe { alloc(layout) as *mut f32 };

        if ptr.is_null() {
            panic!("Memoria insuficiente para el tensor");
        }

        unsafe {
            for i in 0..len {
                ptr.add(i).write(0.0);
            }
        }

        Self { ptr, len, layout }
    }

    /// Construye un tensor copiando los datos de un slice seguro de Rust.
    pub fn from_slice(datos: &[f32]) -> Self {
        let tensor = Self::new(datos.len());
        unsafe {
            // Copia eficiente de bloques de memoria a nivel de punteros:
            std::ptr::copy_nonoverlapping(datos.as_ptr(), tensor.ptr, datos.len());
        }
        tensor
    }

    /// Retorna una vista segura como slice inmutable.
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Retorna una vista segura como slice mutable.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Retorna la longitud del tensor.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Verifica si el tensor está vacío.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// 🚀 Operación Vectorial Acelerada (Producto Punto / Dot Product):
    /// Utiliza aritmética de punteros crudos sin verificación de límites (bounds checking),
    /// tal como hacen los kernels de cuBLAS / TensorRT.
    pub fn producto_punto_acelerado(&self, otro: &TensorContinuo) -> f32 {
        assert_eq!(self.len, otro.len, "Dimensiones incompatibles para producto punto");

        let mut acumulador: f32 = 0.0;
        let p1 = self.ptr;
        let p2 = otro.ptr;

        unsafe {
            for i in 0..self.len {
                let v1 = *p1.add(i);
                let v2 = *p2.add(i);
                acumulador += v1 * v2;
            }
        }

        acumulador
    }
}

// 💡 DESTRUCTOR AUTOMÁTICO RAII:
impl Drop for TensorContinuo {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
    }
}

fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Tensor Continuo y Producto Punto con Punteros");
    println!("============================================================\n");

    let t1 = TensorContinuo::from_slice(&[1.0, 2.0, 3.0, 4.0]);
    let t2 = TensorContinuo::from_slice(&[0.5, 1.0, 1.5, 2.0]);

    println!("Tensor 1: {:?}", t1.as_slice());
    println!("Tensor 2: {:?}", t2.as_slice());

    println!("\nCalculando producto punto acelerado con punteros directos...");
    let resultado = t1.producto_punto_acelerado(&t2);

    println!("Resultado: {:.2}", resultado);
    println!("Esperado : 1*0.5 + 2*1 + 3*1.5 + 4*2 = 0.5 + 2.0 + 4.5 + 8.0 = 15.00");

    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_from_slice_correcto() {
        let datos = [1.5, 2.5, 3.5];
        let tensor = TensorContinuo::from_slice(&datos);
        assert_eq!(tensor.as_slice(), &datos);
        assert_eq!(tensor.len(), 3);
    }

    #[test]
    fn test_producto_punto_acelerado() {
        let t1 = TensorContinuo::from_slice(&[2.0, 3.0]);
        let t2 = TensorContinuo::from_slice(&[4.0, 5.0]);
        let dot = t1.producto_punto_acelerado(&t2);
        assert_eq!(dot, 2.0 * 4.0 + 3.0 * 5.0); // 8 + 15 = 23
    }

    #[test]
    #[should_panic(expected = "Dimensiones incompatibles")]
    fn test_producto_punto_dimensiones_incompatibles_falla() {
        let t1 = TensorContinuo::from_slice(&[1.0, 2.0]);
        let t2 = TensorContinuo::from_slice(&[1.0]);
        let _ = t1.producto_punto_acelerado(&t2);
    }
}
