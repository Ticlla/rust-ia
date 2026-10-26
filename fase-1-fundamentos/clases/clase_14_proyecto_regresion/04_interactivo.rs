//! ============================================================================
//! 🤖 CLASE 14: VERSIÓN INTERACTIVA - Regresión Lineal desde Cero
//! ============================================================================
//!
//! Este programa interactivo te permite:
//! 1. Definir tu propia función objetivo secreta (y = w_real * x + b_real).
//! 2. Ajustar hiperparámetros (Learning Rate y Épocas).
//! 3. Ver el entrenamiento en tiempo real con barra de progreso.
//! 4. Probar predicciones interactivas ingresando valores de X por teclado.
//! 5. Simular qué pasa cuando el Learning Rate es explosivo (Divergencia).
//!
//! Ejecutar con:
//!     cargo run --bin c14_interactivo

use std::io::{self, Write};

fn main() {
    println!("========================================================");
    println!("🤖 LABORATORIO INTERACTIVO: REGRESIÓN LINEAL EN RUST");
    println!("========================================================\n");

    loop {
        println!("Selecciona una opción:");
        println!("  1. 🎯 Entrenar y probar predicciones interactivas");
        println!("  2. 💥 Simular Divergencia (¿Qué pasa con un Learning Rate explosivo?)");
        println!("  3. 🚪 Salir");

        let opcion = leer_texto("\nElige una opción (1-3): ");

        match opcion.as_str() {
            "1" => modo_entrenamiento_interactivo(),
            "2" => modo_simulacion_divergencia(),
            "3" | "q" | "exit" => {
                println!("\n👋 ¡Hasta luego! Sigue aprendiendo Rust para IA.");
                break;
            }
            _ => println!("❌ Opción inválida. Intenta nuevamente.\n"),
        }
    }
}

// =============================================================================
// MODO 1: ENTRENAMIENTO Y PREDICCIÓN INTERACTIVA
// =============================================================================

fn modo_entrenamiento_interactivo() {
    println!("\n--------------------------------------------------------");
    println!("⚙️ CONFIGURACIÓN DEL MODELO Y DATASET");
    println!("--------------------------------------------------------");

    // 1. Elegir la relación objetivo oculta
    println!("Definamos la relación oculta que la IA debe aprender: y = W * x + B");
    let w_real = leer_f32("  Ingrese el Peso objetivo (W, default 3.0): ", 3.0);
    let b_real = leer_f32("  Ingrese el Sesgo objetivo (B, default 2.0): ", 2.0);

    // 2. Elegir hiperparámetros
    let lr = leer_f32("\n  Tasa de aprendizaje (Learning Rate, default 0.01): ", 0.01);
    let epocas = leer_usize("  Cantidad de épocas de entrenamiento (default 1000): ", 1000);

    // 3. Generar dataset sintético basado en W y B
    let mut x_train = Vec::new();
    let mut y_train = Vec::new();
    for i in 1..=8 {
        let x = i as f32;
        let y = w_real * x + b_real;
        x_train.push([x]);
        y_train.push(y);
    }

    println!("\nDataset generado (8 muestras):");
    for (i, (x, &y)) in x_train.iter().zip(y_train.iter()).enumerate() {
        println!("  Muestra {}: x = {:>4.1}  -->  y = {:>6.2}", i + 1, x[0], y);
    }

    // 4. Entrenar el modelo
    println!("\n🚀 Iniciando entrenamiento con Gradiente Descendente...");
    let mut modelo = RegresionLineal::new(lr, epocas);

    let n = x_train.len() as f32;
    for epoca in 1..=epocas {
        let mut grad_w = 0.0;
        let mut grad_b = 0.0;

        for (x, &y) in x_train.iter().zip(y_train.iter()) {
            let pred = modelo.predecir(x);
            let error = pred - y;
            grad_w += (2.0 / n) * error * x[0];
            grad_b += (2.0 / n) * error;
        }

        modelo.peso -= modelo.lr * grad_w;
        modelo.sesgo -= modelo.lr * grad_b;

        // Visualización de progreso cada 100 épocas o última época
        if epoca % 100 == 0 || epoca == epocas {
            let mse = modelo.evaluar_mse(&x_train, &y_train);
            let progreso = (epoca as f32 / epocas as f32 * 20.0) as usize;
            let barra = "=".repeat(progreso);
            let espacios = " ".repeat(20 - progreso);
            print!(
                "\r  Época [{}{}] {:>4}/{} | MSE: {:>8.5} | w: {:>6.3} | b: {:>6.3}",
                barra, espacios, epoca, epocas, mse, modelo.peso, modelo.sesgo
            );
            io::stdout().flush().unwrap();
        }
    }

    println!("\n\n✅ ¡Entrenamiento completado!");
    println!("  • Peso aprendido (w) : {:.4} (Objetivo real: {:.4})", modelo.peso, w_real);
    println!("  • Sesgo aprendido (b): {:.4} (Objetivo real: {:.4})", modelo.sesgo, b_real);

    // 5. Consola de inferencia en vivo
    println!("\n--------------------------------------------------------");
    println!("🔮 MODO PREDICCIÓN EN VIVO");
    println!("Ingresa valores de 'x' para que el modelo haga inferencias.");
    println!("(Escribe 'q' o presiona Enter sin texto para volver al menú)");
    println!("--------------------------------------------------------");

    loop {
        let entrada = leer_texto("\n👉 Ingresa un valor de x: ");
        if entrada.is_empty() || entrada == "q" || entrada == "salir" {
            break;
        }

        match entrada.parse::<f32>() {
            Ok(x_val) => {
                let pred = modelo.predecir(&[x_val]);
                let real = w_real * x_val + b_real;
                let error_abs = (pred - real).abs();
                println!("   ---------------------------------------------");
                println!("   Predicción de la IA  : {:.4}", pred);
                println!("   Valor Real Teórico   : {:.4}", real);
                println!("   Error Absoluto       : {:.6}", error_abs);
                println!("   ---------------------------------------------");
            }
            Err(_) => println!("   ❌ Entrada no válida. Ingresa un número (ej. 10.5)"),
        }
    }
    println!();
}

// =============================================================================
// MODO 2: SIMULACIÓN DE DIVERGENCIA (LEARNING RATE EXPLOSIVO)
// =============================================================================

fn modo_simulacion_divergencia() {
    println!("\n--------------------------------------------------------");
    println!("💥 LABORATORIO DE DIVERGENCIA NUMÉRICA");
    println!("--------------------------------------------------------");
    println!("¿Qué ocurre si el Learning Rate es demasiado grande?");
    println!("En lugar de descender hacia el mínimo de la curva de error,");
    println!("el modelo dará saltos gigantescos que hacen explotar los pesos a infinito o NaN.\n");

    let lr_explosivo = leer_f32("Ingrese un Learning Rate alto para experimentar (ej. 0.25): ", 0.25);

    let x_train = [[1.0], [2.0], [3.0], [4.0]];
    let y_train = [3.0, 5.0, 7.0, 9.0];

    let mut modelo = RegresionLineal::new(lr_explosivo, 15);
    println!("\nEntrenando 15 épocas con LR = {}...", lr_explosivo);

    let n = x_train.len() as f32;
    for epoca in 1..=15 {
        let mut grad_w = 0.0;
        let mut grad_b = 0.0;

        for (x, &y) in x_train.iter().zip(y_train.iter()) {
            let pred = modelo.predecir(x);
            let error = pred - y;
            grad_w += (2.0 / n) * error * x[0];
            grad_b += (2.0 / n) * error;
        }

        modelo.peso -= modelo.lr * grad_w;
        modelo.sesgo -= modelo.lr * grad_b;

        let mse = modelo.evaluar_mse(&x_train, &y_train);
        println!(
            "  Época {:>2} -> MSE: {:>14.2} | Peso: {:>12.2} | Sesgo: {:>12.2}",
            epoca, mse, modelo.peso, modelo.sesgo
        );

        if mse.is_nan() || mse.is_infinite() || mse > 1e10 {
            println!("\n💥 ¡EXPLOSIÓN NUMÉRICA DETECTADA! (Divergencia)");
            println!("El gradiente creció tanto que se desbordó la capacidad del tipo f32.");
            println!("💡 Por eso en Deep Learning usamos schedulers o normalización (Batch Norm).\n");
            return;
        }
    }
    println!();
}

// =============================================================================
// ESTRUCTURA DEL MODELO
// =============================================================================

struct RegresionLineal {
    peso: f32,
    sesgo: f32,
    lr: f32,
    #[allow(dead_code)]
    epocas: usize,
}

impl RegresionLineal {
    fn new(lr: f32, epocas: usize) -> Self {
        Self {
            peso: 0.0,
            sesgo: 0.0,
            lr,
            epocas,
        }
    }

    fn predecir(&self, x: &[f32]) -> f32 {
        x[0] * self.peso + self.sesgo
    }

    fn evaluar_mse(&self, x: &[[f32; 1]], y: &[f32]) -> f32 {
        let n = x.len() as f32;
        let suma: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, &yi)| {
                let err = self.predecir(xi) - yi;
                err * err
            })
            .sum();
        suma / n
    }
}

// =============================================================================
// FUNCIONES AUXILIARES DE ENTRADA / SALIDA POR CONSOLA
// =============================================================================

fn leer_texto(prompt: &str) -> String {
    print!("{}", prompt);
    io::stdout().flush().unwrap();
    let mut buf = String::new();
    match io::stdin().read_line(&mut buf) {
        Ok(0) => {
            // Fin de archivo (EOF): salir de forma segura
            "3".to_string()
        }
        Ok(_) => buf.trim().to_string(),
        Err(_) => "3".to_string(),
    }
}

fn leer_f32(prompt: &str, defecto: f32) -> f32 {
    let entrada = leer_texto(prompt);
    if entrada.is_empty() {
        defecto
    } else {
        entrada.parse::<f32>().unwrap_or(defecto)
    }
}

fn leer_usize(prompt: &str, defecto: usize) -> usize {
    let entrada = leer_texto(prompt);
    if entrada.is_empty() {
        defecto
    } else {
        entrada.parse::<usize>().unwrap_or(defecto)
    }
}
