//! ============================================================================
//! 🛠️ FASE 2 - CLASE 08: Laboratorio de Errores Comunes con Axum
//! ============================================================================
//!
//! Este laboratorio explora los 3 errores de compilación más desconcertantes
//! al programar APIs web en Rust con Axum.
//!
//! Para ejecutar este laboratorio:
//!     cargo run --bin c08_errores

fn main() {
    println!("============================================================");
    println!("🛠️ LABORATORIO DE ERRORES: Servidores Web con Axum");
    println!("============================================================\n");

    // =========================================================================
    // ERROR COMÚN 1: El orden de los Extractores (Json consume el body)
    // =========================================================================
    println!("--- Error 1: El orden de los Extractores en el Handler ---");
    println!("Si intentas escribir:");
    println!("    async fn handler_malo(");
    println!("        Json(payload): Json<Datos>, // ❌ Consume el body primero!");
    println!("        State(state): State<AppState>,");
    println!("    )");
    println!("\n¿Por qué falla el compilador?");
    println!("Porque 'Json<T>' consume el flujo de bytes de la petición HTTP (Request Body).");
    println!("En Axum existe una regla estricta: cualquier extractor que consuma el body");
    println!("DEBE SER EL ÚLTIMO PARÁMETRO de la función.");
    println!("👉 REGLA DE ORO: Pon siempre 'State' y 'Path' al principio y 'Json' al final:");
    println!("    async fn handler_bueno(State(state): State<...>, Json(payload): Json<...>) // ✅\n");

    // =========================================================================
    // ERROR COMÚN 2: Discrepancia en el tipo de State
    // =========================================================================
    println!("--- Error 2: Desajuste de tipos con .with_state() ---");
    println!("Si configuras el router con:");
    println!("    let state = Arc::new(AppState {{ ... }});");
    println!("    Router::new().route(\"/\", get(mi_handler)).with_state(state);");
    println!("Pero en tu handler pides:");
    println!("    async fn mi_handler(State(state): State<AppState>) // ❌ Esperaba Arc<AppState>!");
    println!("\n¿Qué dice el compilador de Rust?");
    println!("    error[E0277]: the trait `Handler<_, _>` is not implemented for `fn(State<AppState>)`");
    println!("👉 SOLUCIÓN: Asegúrate de que el tipo exacto en 'State<T>' coincida con lo pasado a .with_state().\n");

    // =========================================================================
    // ERROR COMÚN 3: Retornar tipos que no implementan IntoResponse
    // =========================================================================
    println!("--- Error 3: Retornar structs sin envolver en Json o IntoResponse ---");
    println!("Si intentas escribir:");
    println!("    async fn handler_crudo() -> MiStructPrediccion {{ ... }} // ❌ ERROR");
    println!("\n¿Por qué falla?");
    println!("Porque Axum no sabe cómo enviar 'MiStructPrediccion' a través del protocolo HTTP");
    println!("a menos que lo envuelvas en 'Json(MiStructPrediccion)' o implementes 'IntoResponse'.");
    println!("👉 SOLUCIÓN: Retorna siempre 'Json(mi_struct)' o '(StatusCode::OK, Json(mi_struct))'.\n");

    println!("✅ Todos los casos de error en Axum comprendidos.");
}
