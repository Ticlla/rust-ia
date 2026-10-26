//! ============================================================================
//! 🦀 CLASE 14 / PROYECTO FINAL: Regresión Lineal desde Cero en Rust
//! ============================================================================
//!
//! ¡Felicidades por llegar aquí! Este proyecto consolida TODO lo aprendido
//! a lo largo de las 13 clases de la Fase 1: Fundamentos.
//!
//! Conceptos aplicados:
//! 1. Variables, mutabilidad y tipos (Clases 01 y 02).
//! 2. Funciones y control de flujo en épocas de entrenamiento (Clases 03 y 04).
//! 3. Ownership, Borrowing (&) y Slices (&[f32]) para cero copias de datos (Clases 05, 06 y 07).
//! 4. Structs y métodos asociados con 'impl' y '&mut self' (Clase 08).
//! 5. Enums y manejo robusto de errores con Result<T, E> (Clases 09 y 10).
//! 6. Iteradores funcionales con .zip(), .map() y .sum() (Clase 11).
//! 7. Traits y abstracción polimórfica de modelos de IA (Clase 12).
//! 8. Seguridad de referencias en tiempo de compilación (Clase 13).
//!
//! Para ejecutar este proyecto:
//!     cargo run --bin clase_14_proyecto_regresion_lineal

fn main() {
    println!("========================================================");
    println!("🚀 CLASE 14 (CHECKPOINT FASE 1): REGRESIÓN LINEAL");
    println!("========================================================\n");

    // =========================================================================
    // 1. DATASET DE ENTRENAMIENTO (Relación real subyacente: y = 3.0 * x + 2.0)
    // =========================================================================
    println!("--- 1. Preparando el Dataset de Entrenamiento ---");

    // Datos de entrada (X): 1 feature por muestra
    let x_train: [[f32; 1]; 6] = [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ];

    // Etiquetas reales (Y): y = 3x + 2
    let y_train: [f32; 6] = [
        5.0,   // 3(1) + 2
        8.0,   // 3(2) + 2
        11.0,  // 3(3) + 2
        14.0,  // 3(4) + 2
        17.0,  // 3(5) + 2
        20.0,  // 3(6) + 2
    ];

    println!("Dataset: 6 muestras. Relación objetivo: y = 3.0 * x + 2.0\n");

    // =========================================================================
    // 2. INICIALIZACIÓN DEL MODELO
    // =========================================================================
    println!("--- 2. Inicializando el Modelo ---");

    // Tasa de aprendizaje (learning rate = 0.01) y 1000 épocas
    let mut modelo = RegresionLineal::new(0.01, 1000);

    println!("Arquitectura: {}", modelo.nombre());
    println!("Pesos iniciales : peso = {:.4}, sesgo = {:.4}", modelo.peso, modelo.sesgo);

    // Evaluamos el error antes de entrenar
    match modelo.evaluar_mse(&x_train, &y_train) {
        Ok(mse_inicial) => println!("Error Inicial (MSE): {:.4}\n", mse_inicial),
        Err(e) => eprintln!("Error al evaluar: {}", e),
    }

    // =========================================================================
    // 3. ENTRENAMIENTO (GRADIENTE DESCENDENTE)
    // =========================================================================
    println!("--- 3. Iniciando Entrenamiento con Gradiente Descendente ---");

    match modelo.entrenar(&x_train, &y_train) {
        Ok(()) => println!("\n✅ Entrenamiento completado con éxito!"),
        Err(e) => {
            eprintln!("❌ Error durante el entrenamiento: {}", e);
            return;
        }
    }

    println!("\nParámetros aprendidos:");
    println!("  Peso (w) estimado : {:.4} (Esperado: ~3.0)", modelo.peso);
    println!("  Sesgo (b) estimado: {:.4} (Esperado: ~2.0)", modelo.sesgo);

    match modelo.evaluar_mse(&x_train, &y_train) {
        Ok(mse_final) => println!("  Error Final (MSE) : {:.6}\n", mse_final),
        Err(e) => eprintln!("Error al evaluar: {}", e),
    }

    // =========================================================================
    // 4. INFERENCIA CON NUEVOS DATOS (PREDICCIONES)
    // =========================================================================
    println!("--- 4. Inferencia con Datos No Vistos ---");

    let nuevos_datos = [7.0, 10.0, 15.0];
    for &nuevo_x in &nuevos_datos {
        let pred = modelo.predecir(&[nuevo_x]);
        let valor_teorico = 3.0 * nuevo_x + 2.0;
        println!(
            "  x = {:5.1} -> Predicción: {:6.2} | Real teórico: {:6.2}",
            nuevo_x, pred, valor_teorico
        );
    }

    // =========================================================================
    // 5. DEMOSTRACIÓN DE ROBUSTEZ: MANEJO DE ERRORES (Result)
    // =========================================================================
    println!("\n--- 5. Prueba de Robustez contra Datos Inválidos ---");

    let x_invalido: [[f32; 1]; 3] = [[1.0], [2.0], [3.0]];
    let y_invalido: [f32; 2] = [5.0, 8.0]; // ¡Solo 2 etiquetas para 3 muestras!

    println!("Intentando entrenar con dimensiones incompatibles (3 muestras vs 2 etiquetas)...");
    match modelo.entrenar(&x_invalido, &y_invalido) {
        Ok(()) => println!("Entrenamiento inesperado!"),
        Err(err) => println!("🛡️ Rust capturó el error elegantemente: {}", err),
    }

    println!("\n🎉 ¡Felicidades! Has completado exitosamente la Fase 1.");
}

// =============================================================================
// DEFINICIÓN DEL TRAIT COMÚN PARA MODELOS
// =============================================================================

/// Trait que abstrae el comportamiento de cualquier modelo de inferencia en IA.
pub trait Modelo {
    /// Inferencia: dado un vector de características, retorna la predicción.
    fn predecir(&self, x: &[f32]) -> f32;

    /// Nombre descriptivo del modelo.
    fn nombre(&self) -> &str;

    /// Método por defecto: Calcula el Error Cuadrático Medio (Mean Squared Error - MSE).
    /// Utiliza iteradores funcionales (.zip, .map, .sum) para máxima eficiencia.
    fn evaluar_mse(&self, entradas: &[[f32; 1]], etiquetas: &[f32]) -> Result<f32, String> {
        if entradas.len() != etiquetas.len() {
            return Err(format!(
                "Dimensiones incompatibles: {} muestras de entrada vs {} etiquetas",
                entradas.len(),
                etiquetas.len()
            ));
        }

        if entradas.is_empty() {
            return Err("El dataset no puede estar vacío".to_string());
        }

        // Iterador funcional sin clonar datos:
        let suma_error_cuadratico: f32 = entradas
            .iter()
            .zip(etiquetas.iter())
            .map(|(x, &y_real)| {
                let y_pred = self.predecir(x);
                let residuo = y_pred - y_real;
                residuo * residuo
            })
            .sum();

        Ok(suma_error_cuadratico / entradas.len() as f32)
    }
}

// =============================================================================
// STRUCT DEL MODELO: REGRESIÓN LINEAL SIMPLE
// =============================================================================

/// Representa el modelo lineal y = w * x + b
pub struct RegresionLineal {
    pub peso: f32,           // w (weight)
    pub sesgo: f32,          // b (bias)
    pub learning_rate: f32,  // Tasa de aprendizaje (eta)
    pub epocas: usize,       // Número de iteraciones
}

impl RegresionLineal {
    /// Constructor: Inicializa los pesos con valores iniciales
    pub fn new(learning_rate: f32, epocas: usize) -> Self {
        Self {
            peso: 0.0,
            sesgo: 0.0,
            learning_rate,
            epocas,
        }
    }

    /// Entrena el modelo usando el algoritmo de Gradiente Descendente.
    /// Recibe préstamos (&) para evitar duplicar matrices en RAM.
    /// Recibe '&mut self' porque modificará 'peso' y 'sesgo'.
    pub fn entrenar(&mut self, x: &[[f32; 1]], y: &[f32]) -> Result<(), String> {
        if x.len() != y.len() {
            return Err(format!(
                "Error en entrenamiento: X tiene {} muestras pero Y tiene {}",
                x.len(),
                y.len()
            ));
        }

        if x.is_empty() {
            return Err("El dataset de entrenamiento está vacío".to_string());
        }

        let n = x.len() as f32;

        for epoca in 1..=self.epocas {
            let mut gradiente_peso = 0.0;
            let mut gradiente_sesgo = 0.0;

            // Paso 1: Acumular gradientes sobre todas las muestras
            // Error: y_pred - y_real
            for (muestra, &y_real) in x.iter().zip(y.iter()) {
                let x_val = muestra[0];
                let y_pred = self.predecir(muestra);
                let error = y_pred - y_real;

                gradiente_peso += (2.0 / n) * error * x_val;
                gradiente_sesgo += (2.0 / n) * error;
            }

            // Paso 2: Actualizar pesos en dirección opuesta al gradiente
            self.peso -= self.learning_rate * gradiente_peso;
            self.sesgo -= self.learning_rate * gradiente_sesgo;

            // Paso 3: Monitorear el progreso cada 200 épocas
            if epoca % 200 == 0 || epoca == 1 {
                let mse = self.evaluar_mse(x, y)?;
                println!(
                    "  [Época {:>4}/{}] -> MSE: {:>8.4} | Peso: {:>6.4} | Sesgo: {:>6.4}",
                    epoca, self.epocas, mse, self.peso, self.sesgo
                );
            }
        }

        Ok(())
    }
}

// =============================================================================
// IMPLEMENTACIÓN DEL TRAIT 'Modelo' PARA 'RegresionLineal'
// =============================================================================

impl Modelo for RegresionLineal {
    fn predecir(&self, x: &[f32]) -> f32 {
        x[0] * self.peso + self.sesgo
    }

    fn nombre(&self) -> &str {
        "Regresión Lineal Simple (Gradiente Descendente)"
    }
}
