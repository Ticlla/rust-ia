//! ============================================================================
//! 🦀 FASE 2 - CLASE 09: Punteros Crudos y `unsafe` Rust (C / CUDA) en IA
//! ============================================================================
//!
//! En esta clase aprenderemos:
//! 1. Qué es y qué NO es `unsafe`: Desbloquear los 5 superpoderes de bajo nivel.
//! 2. Punteros crudos: `*const T` (inmutable) y `*mut T` (mutable).
//! 3. El patrón fundamental en IA: El "Safe Wrapper" (Envoltorio Seguro).
//! 4. Cómo interactúa Rust con librerías externas en C y CUDA (FFI).
//! 5. Liberación RAII con `Drop` para evitar Memory Leaks en tensores manuales.
//!
//! Para ejecutar este archivo:
//!     cargo run --bin c09_teoria

use std::alloc::{alloc, dealloc, Layout};

// =============================================================================
// 1. SIMULACIÓN DE UNA LIBRERÍA EXTERNA EN C / KERNEL DE CUDA (FFI)
// =============================================================================

/// Simula una función escrita en C o un kernel de NVIDIA CUDA (como cuBLAS):
/// Multiplica un vector de entrada por un escalar y guarda el resultado: Y = alpha * X
/// Las funciones externas en C son intrínsecamente 'unsafe' porque reciben punteros crudos.
unsafe extern "C" fn cuda_scale_tensor_mock(
    n: usize,
    alpha: f32,
    x: *const f32, // Puntero crudo de solo lectura a la memoria
    y: *mut f32,   // Puntero crudo de escritura en memoria
) {
    // En CUDA real, esto correría en miles de hilos de la GPU:
    unsafe {
        for i in 0..n {
            let val_x = *x.add(i); // Desreferenciamos el puntero con aritmética de punteros
            *y.add(i) = alpha * val_x;
        }
    }
}

// =============================================================================
// 2. EL PATRÓN "SAFE WRAPPER" (ENVOLTORIO SEGURO PARA TENSORES)
// =============================================================================

/// Un búfer de memoria contigua para tensores de IA asignado manualmente en el Heap.
/// Encapsula todo el código 'unsafe' para que el usuario final use una API 100% segura.
pub struct TensorBuffer {
    ptr: *mut f32,
    len: usize,
    layout: Layout,
}

impl TensorBuffer {
    /// Asigna memoria continua y alineada para 'len' números decimales (f32).
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "El tensor no puede tener tamaño 0");

        // Definimos la disposición en memoria (tamaño y alineación de 4 bytes para f32):
        let layout = Layout::array::<f32>(len).expect("Layout de memoria inválido");

        // 💡 'alloc' es una función de bajo nivel que pide bytes al sistema:
        let ptr = unsafe { alloc(layout) as *mut f32 };

        if ptr.is_null() {
            panic!("Fallo al asignar memoria física para el tensor");
        }

        // Inicializamos los valores en 0.0:
        unsafe {
            for i in 0..len {
                ptr.add(i).write(0.0);
            }
        }

        Self { ptr, len, layout }
    }

    /// Retorna una referencia de solo lectura como un slice seguro de Rust: &[f32].
    pub fn as_slice(&self) -> &[f32] {
        // 💡 Convertir un puntero crudo verificado a un slice seguro:
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Retorna una referencia mutable como un slice seguro de Rust: &mut [f32].
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Llama al kernel de CUDA / C de forma segura:
    pub fn escalar_con_cuda(&mut self, alpha: f32) {
        unsafe {
            // Pasamos los punteros crudos al kernel:
            cuda_scale_tensor_mock(self.len, alpha, self.ptr, self.ptr);
        }
    }
}

// 💡 DESTRUCTOR AUTOMÁTICO (RAII):
// Cuando el 'TensorBuffer' sale de ámbito, 'Drop' libera la memoria automáticamente.
// ¡Nunca hay fugas de memoria (Memory Leaks)!
impl Drop for TensorBuffer {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr as *mut u8, self.layout);
        }
        println!("  [RAII] 🗑️ Memoria física del tensor liberada con éxito.");
    }
}

// =============================================================================
// 3. MAIN: DEMOSTRACIÓN DE BAJO NIVEL
// =============================================================================

fn main() {
    println!("============================================================");
    println!("⚡ FASE 2 - CLASE 09: Punteros Crudos y unsafe Rust (C/CUDA)");
    println!("============================================================\n");

    // =========================================================================
    // PARTE A: CREACIÓN Y DESREFERENCIACIÓN BÁSICA DE PUNTEROS CRUDOS
    // =========================================================================
    println!("--- 1. Punteros Crudos: *const y *mut en Rust ---");

    let mut peso_neuronal: f32 = 0.85;

    // Crear punteros crudos es 100% SEGURO (no requiere bloque unsafe):
    let ptr_inmutable: *const f32 = &peso_neuronal as *const f32;
    let ptr_mutable: *mut f32 = &mut peso_neuronal as *mut f32;

    println!("Dirección física de memoria: {:p}", ptr_inmutable);

    // 💡 DESREFERENCIAR el puntero (leer o escribir en la dirección) SÍ requiere 'unsafe':
    unsafe {
        println!("Valor leído a través de *const: {:.2}", *ptr_inmutable);

        // Modificamos el valor directamente en el silicio:
        *ptr_mutable = 0.99;
        println!("Valor modificado a través de *mut: {:.2}", *ptr_mutable);
    }
    println!("Valor en la variable original de Rust: {:.2}\n", peso_neuronal);

    // =========================================================================
    // PARTE B: EL ENVOLTORIO SEGURO EN ACCIÓN
    // =========================================================================
    println!("--- 2. Safe Wrapper para Tensores con Kernel CUDA Simulado ---");

    {
        let mut tensor = TensorBuffer::new(4);

        // Llenamos datos con la API segura:
        tensor.as_mut_slice().copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        println!("Tensor inicial antes de CUDA: {:?}", tensor.as_slice());

        // Multiplicamos por 10.0 llamando a la función estilo C/CUDA:
        println!("Llamando al kernel CUDA (Y = 10.0 * X)...");
        tensor.escalar_con_cuda(10.0);

        println!("Tensor resultante tras CUDA: {:?}", tensor.as_slice());
        println!("Esperado: [10.0, 20.0, 30.0, 40.0]");
        println!("Saliendo del ámbito de bloque {{ }}...");
    } // 👈 ¡Aquí se ejecuta 'Drop' automáticamente y libera los bytes!

    println!("\n✅ Fin de la Clase 09 de bajo nivel.");
}
