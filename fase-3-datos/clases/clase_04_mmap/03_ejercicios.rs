//! ==============================================================================
//! CLASE 04: Ejercicios Prácticos con Memoria Mapeada (`mmap`)
//! ==============================================================================
//!
//! En este laboratorio construirás un cargador de modelos de grado industrial:
//! 1. Crear un contenedor seguro para mantener vivo el archivo `Mmap`.
//! 2. Extraer tensores directamente del disco sin copiarlos a la RAM.
//! 3. Ejecutar una inferencia tensorial completa (Matmul de la Clase 01)
//!    usando los pesos mapeados en disco.

use memmap2::Mmap;
use ndarray::{Array1, Array2};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use safetensors::serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn main() {
    println!("============================================================");
    println!("🧪 EJERCICIOS CLASE 04: Inferencia con Memoria Mapeada (mmap)");
    println!("============================================================\n");
    println!("Ejecuta 'cargo test --bin c04_ejercicios' para evaluar tus soluciones.");
}

/// ----------------------------------------------------------------------------
/// ESTRUCTURA: Contenedor de Modelo Mapeado en Memoria
/// ----------------------------------------------------------------------------
pub struct ModeloMapeado {
    _mmap: Mmap,
    _archivo: File,
}

impl ModeloMapeado {
    pub fn abrir<P: AsRef<Path>>(ruta: P) -> Result<Self, Box<dyn std::error::Error>> {
        let archivo = File::open(ruta)?;
        let mmap = unsafe { Mmap::map(&archivo)? };
        Ok(Self {
            _mmap: mmap,
            _archivo: archivo,
        })
    }

    pub fn data(&self) -> &[u8] {
        &self._mmap
    }
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 1: Leer Tensor 2D directamente del archivo mapeado
/// ----------------------------------------------------------------------------
pub fn leer_matriz_mmap<'a>(
    modelo: &'a ModeloMapeado,
    nombre_tensor: &str,
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
    let st = SafeTensors::deserialize(modelo.data())?;
    let t = st.tensor(nombre_tensor)?;

    let shape = t.shape();
    if shape.len() != 2 {
        return Err("El tensor debe ser de 2 dimensiones".into());
    }

    let bytes = t.data();
    let floats: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    Ok(Array2::from_shape_vec((shape[0], shape[1]), floats.to_vec())?)
}

/// ----------------------------------------------------------------------------
/// EJERCICIO 2: Inferencia Matmul Directa (Y = X · W) usando Pesos en Disco
/// ----------------------------------------------------------------------------
pub fn inferir_con_pesos_en_disco(
    entrada: &Array1<f32>,
    modelo: &ModeloMapeado,
    nombre_pesos: &str,
) -> Result<f32, Box<dyn std::error::Error>> {
    let st = SafeTensors::deserialize(modelo.data())?;
    let t = st.tensor(nombre_pesos)?;

    let bytes = t.data();
    let pesos_disco: &[f32] = unsafe {
        std::slice::from_raw_parts(
            bytes.as_ptr() as *const f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    if pesos_disco.len() != entrada.len() {
        return Err("Dimensiones incompatibles".into());
    }

    // Producto punto directo contra la memoria del disco SSD (Page Cache del kernel):
    let mut dot_product = 0.0f32;
    for i in 0..pesos_disco.len() {
        dot_product += entrada[i] * pesos_disco[i];
    }

    Ok(dot_product)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crear_archivo_safetensors_dummy(ruta: &Path) {
        let pesos: Vec<f32> = vec![2.0, -1.0, 0.5]; // Vector 1x3
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                pesos.as_ptr() as *const u8,
                pesos.len() * std::mem::size_of::<f32>(),
            )
        };

        let tv = TensorView::new(Dtype::F32, vec![1, 3], bytes).unwrap();
        let mut mapa = HashMap::new();
        mapa.insert("linear.weight", tv);

        let data = serialize(&mapa, None).unwrap();
        let mut f = File::create(ruta).unwrap();
        f.write_all(&data).unwrap();
    }

    #[test]
    fn test_modelo_mmap_lectura_e_inferencia() {
        let ruta = std::env::temp_dir().join("test_mmap_inferencia.safetensors");
        crear_archivo_safetensors_dummy(&ruta);

        let modelo = ModeloMapeado::abrir(&ruta).unwrap();

        // 1. Validar lectura de tensor
        let matriz = leer_matriz_mmap(&modelo, "linear.weight").unwrap();
        assert_eq!(matriz.shape(), &[1, 3]);
        assert_eq!(matriz[[0, 0]], 2.0);
        assert_eq!(matriz[[0, 1]], -1.0);
        assert_eq!(matriz[[0, 2]], 0.5);

        // 2. Inferencia directa sobre disco: X = [10.0, 5.0, 2.0]
        // y = 10*2.0 + 5*(-1.0) + 2*0.5 = 20.0 - 5.0 + 1.0 = 16.0
        let x = ndarray::array![10.0f32, 5.0, 2.0];
        let resultado = inferir_con_pesos_en_disco(&x, &modelo, "linear.weight").unwrap();
        assert!((resultado - 16.0).abs() < 1e-5);

        let _ = std::fs::remove_file(&ruta);
    }
}
