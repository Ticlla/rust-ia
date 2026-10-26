//! ==============================================================================
//! CLASE 04: Memoria Mapeada (`mmap`) para Modelos Gigantes (LLMs)
//! ==============================================================================
//!
//! En Inteligencia Artificial moderna, los modelos Open Weights (Llama 3, Qwen,
//! DeepSeek, Mistral) pesan entre 15 GB y más de 140 GB.
//!
//! Si intentas abrir un archivo de 70 GB con `std::fs::read("modelo.safetensors")`,
//! tu computadora colapsará al instante con un pánico "Out Of Memory" (OOM).
//!
//! La técnica secreta que usan motores como `vLLM`, `llama.cpp` y `Candle` para
//! arrancar modelos gigantes en 0.001 segundos se llama **Memoria Mapeada (`mmap`)**.
//!
//! OBJETIVOS PEDAGÓGICOS:
//! 1. Entender la llamada al kernel de Linux `mmap` y la Paginación Virtual.
//! 2. Abrir archivos de modelos de cualquier tamaño en tiempo récord con `memmap2`.
//! 3. El Dúo Invencible: Integrar `mmap` con `safetensors` para inferencia Cero-Copias.

use memmap2::Mmap;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use safetensors::serialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("============================================================");
    println!("⚡ CLASE 04: Memoria Mapeada (mmap) para Modelos Gigantes");
    println!("============================================================\n");

    demostrar_el_problema_de_ram_vs_mmap();
    demostrar_mmap_con_safetensors()?;

    println!("\n✅ Demostración teórica de mmap completada con éxito.");
    Ok(())
}

/// ----------------------------------------------------------------------------
/// 1. ¿CÓMO FUNCIONA MMAP EN EL KERNEL DE LINUX?
/// ----------------------------------------------------------------------------
/// Cuando haces `mmap`:
/// El kernel de Linux NO lee el archivo a la RAM.
/// Solo asigna un rango de direcciones en el "Espacio de Memoria Virtual" de tu proceso
/// que apunta directamente al archivo en tu disco NVMe.
///
/// El tiempo de carga de un archivo de 100 GB con mmap es de menos de 1 MILISEGUNDO.
fn demostrar_el_problema_de_ram_vs_mmap() {
    println!("--- 1. El Concepto de Memoria Mapeada (Virtual Memory Paging) ---");
    println!("❌ std::fs::read(\"modelo_70GB.safetensors\"):");
    println!("   Intenta reservar un Vec<u8> de 70 GB continuos en RAM física.");
    println!("   -> Resultado: Tu laptop se congela y el kernel mata el proceso (OOM Kill).\n");

    println!("⚡ mmap2::Mmap::map(&archivo):");
    println!("   El kernel de Linux solo crea una tabla de punteros virtuales.");
    println!("   Los datos se quedan en el disco SSD y solo se cargan a la RAM en");
    println!("   pequeñas páginas de 4 KB en el instante exacto en que la CPU los toca.");
    println!("   -> Resultado: Arranca en 0.0001 segundos usando casi 0 MB de RAM inicial.\n");
}

/// ----------------------------------------------------------------------------
/// 2. EL DÚO INVENCIBLE: `mmap` + `safetensors`
/// ----------------------------------------------------------------------------
/// Como vimos en la Clase 03, los archivos `.safetensors` guardan los números
/// de los tensores exactamente con el mismo formato que usa la memoria RAM.
///
/// Por eso, al combinar `mmap` con `safetensors`:
/// Puedes mapear un modelo en disco y leer los punteros a sus matrices
/// ¡SIN COPIAR UN SOLO BYTE DE MEMORIA!
fn demostrar_mmap_con_safetensors() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- 2. Demostración en Vivo: Mmap + SafeTensors ---");

    // 1. Creamos un archivo simulado de modelo en disco (.safetensors):
    let ruta_archivo = std::env::temp_dir().join("modelo_mmap_demo.safetensors");
    let pesos_capa: Vec<f32> = vec![0.5, -1.2, 3.4, 8.1, 0.0, 2.5]; // Matriz 2x3
    let bytes_f32: &[u8] = unsafe {
        std::slice::from_raw_parts(
            pesos_capa.as_ptr() as *const u8,
            pesos_capa.len() * std::mem::size_of::<f32>(),
        )
    };

    let tensor_view = TensorView::new(Dtype::F32, vec![2, 3], bytes_f32)?;
    let mut mapa = HashMap::new();
    mapa.insert("transformer.layer0.attention.weight", tensor_view);

    let bytes_safetensors = serialize(&mapa, None)?;
    let mut file_write = File::create(&ruta_archivo)?;
    file_write.write_all(&bytes_safetensors)?;
    file_write.flush()?;

    println!("📁 Modelo guardado en disco: {:?}", ruta_archivo);

    // 2. ABRIR EL MODELO CON MMAP (Cero copias):
    let file = File::open(&ruta_archivo)?;
    
    // 💡 Llamada directa al kernel de Linux:
    let mmap = unsafe { Mmap::map(&file)? };

    println!("⚡ Archivo mapeado en memoria virtual instantáneamente.");
    println!("   Dirección de memoria virtual asignada por Linux: {:p}", mmap.as_ptr());
    println!("   Tamaño total del archivo: {} bytes", mmap.len());

    // 3. DESERIALIZAR CON SAFETENSORS DIRECTAMENTE SOBRE EL MMAP:
    // 'mmap' se comporta como un &[u8] seguro prestado del disco:
    let safetensors = SafeTensors::deserialize(&mmap)?;
    let tensor = safetensors.tensor("transformer.layer0.attention.weight")?;

    println!("\n🔍 Metadatos del Tensor leídos directamente desde el disco:");
    println!("   Shape: {:?}", tensor.shape());
    println!("   Dtype: {:?}", tensor.dtype());

    // 4. LEER VALORES: La CPU solo lee de disco la página requerida (Page Fault transparente):
    let data_bytes = tensor.data();
    let floats_en_disco: &[f32] = unsafe {
        std::slice::from_raw_parts(
            data_bytes.as_ptr() as *const f32,
            data_bytes.len() / std::mem::size_of::<f32>(),
        )
    };

    println!("   Pesos leídos en vivo desde el disco: {:?}", floats_en_disco);
    assert_eq!(floats_en_disco, &[0.5, -1.2, 3.4, 8.1, 0.0, 2.5]);
    println!("✓ Inferencia directa sobre disco completada sin precargar el modelo en RAM.");

    let _ = std::fs::remove_file(&ruta_archivo);
    Ok(())
}
