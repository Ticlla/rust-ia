//! ============================================================================
//! 🎯 FASE 2 - CLASE 04: Retos Prácticos con `RwLock`
//! ============================================================================
//!
//! Ejecutar el reto:
//!     cargo run --bin c04_ejercicios
//!
//! Ejecutar las pruebas unitarias:
//!     cargo test --bin c04_ejercicios

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::thread;

/// Un almacén de embeddings en memoria para un modelo de lenguaje o búsqueda semántica.
/// Miles de consultas de inferencia pueden leer a la vez con `.read()`,
/// mientras que los nuevos embeddings se insertan de forma segura con `.write()`.
#[derive(Clone)]
pub struct CacheEmbeddings {
    tabla: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl CacheEmbeddings {
    /// Crea una nueva caché vacía.
    pub fn new() -> Self {
        Self {
            tabla: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Inserta o actualiza un embedding usando acceso exclusivo de ESCRITURA.
    pub fn insertar(&self, token: String, vector: Vec<f32>) {
        let mut guard = self.tabla.write().unwrap();
        guard.insert(token, vector);
    }

    /// Consulta un embedding usando acceso concurrente de LECTURA.
    pub fn obtener(&self, token: &str) -> Option<Vec<f32>> {
        let guard = self.tabla.read().unwrap();
        guard.get(token).cloned()
    }

    /// Retorna el número de embeddings almacenados.
    pub fn len(&self) -> usize {
        let guard = self.tabla.read().unwrap();
        guard.len()
    }

    /// Verifica si la caché está vacía.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Ejecuta múltiples consultas de inferencia en paralelo repartidas en hilos.
    pub fn consultar_en_paralelo(&self, tokens: Vec<String>) -> Vec<Option<Vec<f32>>> {
        let mut handles = Vec::new();

        for token in tokens {
            let cache_clon = self.clone();
            let h = thread::spawn(move || {
                cache_clon.obtener(&token)
            });
            handles.push(h);
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }
}

impl Default for CacheEmbeddings {
    fn default() -> Self {
        Self::new()
    }
}

fn main() {
    println!("============================================================");
    println!("🎯 RETO PRÁCTICO: Caché de Embeddings Concurrente en IA");
    println!("============================================================\n");

    let cache = CacheEmbeddings::new();

    // 1. Poblamos la caché inicial:
    println!("1. Insertando embeddings iniciales...");
    cache.insertar("rust".to_string(), vec![0.9, 0.1, 0.5]);
    cache.insertar("ia".to_string(), vec![0.3, 0.8, 0.9]);
    cache.insertar("llm".to_string(), vec![0.7, 0.7, 0.2]);

    println!("Total en caché: {} embeddings\n", cache.len());

    // 2. Consultas concurrentes desde múltiples clientes:
    println!("2. Ejecutando consultas de inferencia concurrentes...");
    let consultas = vec![
        "rust".to_string(),
        "ia".to_string(),
        "desconocido".to_string(),
        "llm".to_string(),
    ];

    let resultados = cache.consultar_en_paralelo(consultas);

    for (idx, res) in resultados.iter().enumerate() {
        println!("  Consulta #{idx}: {:?}", res);
    }

    // 3. Actualización concurrente mientras otros leen:
    println!("\n3. Actualizando embedding 'rust' a nueva dimensión...");
    cache.insertar("rust".to_string(), vec![0.99, 0.99, 0.99]);
    println!("Nuevo vector para 'rust': {:?}", cache.obtener("rust"));

    println!("\n✅ Reto ejecutado exitosamente.");
}

// =============================================================================
// PRUEBAS UNITARIAS AUTOMÁTICAS
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lecturas_concurrentes_correctas() {
        let cache = CacheEmbeddings::new();
        cache.insertar("token1".to_string(), vec![1.0, 2.0]);
        cache.insertar("token2".to_string(), vec![3.0, 4.0]);

        let tokens = vec![
            "token1".to_string(),
            "token2".to_string(),
            "token1".to_string(),
            "token_inexistente".to_string(),
        ];

        let resultados = cache.consultar_en_paralelo(tokens);

        assert_eq!(resultados[0], Some(vec![1.0, 2.0]));
        assert_eq!(resultados[1], Some(vec![3.0, 4.0]));
        assert_eq!(resultados[2], Some(vec![1.0, 2.0]));
        assert_eq!(resultados[3], None);
    }

    #[test]
    fn test_escritura_y_lectura_consistentes() {
        let cache = CacheEmbeddings::new();
        assert!(cache.is_empty());

        cache.insertar("alpha".to_string(), vec![0.1]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.obtener("alpha"), Some(vec![0.1]));

        // Sobrescribir:
        cache.insertar("alpha".to_string(), vec![0.9]);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.obtener("alpha"), Some(vec![0.9]));
    }
}
