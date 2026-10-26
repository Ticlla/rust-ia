//! 🦀 CLASE 08: LABORATORIO DE ERRORES COMUNES - Structs
//! Ejecutar con: cargo run --bin c08_errores

struct Capa {
    neuronas: usize,
}

impl Capa {
    // Si olvidamos &mut self y ponemos &self al mutar:
    fn cambiar(&mut self, n: usize) {
        self.neuronas = n;
    }
}

fn main() {
    println!("========================================================");
    println!("⚠️  CLASE 08 [LABORATORIO DE ERRORES]: Structs");
    println!("========================================================\n");

    println!("--- Error 1: Llamar un método &mut self en una instancia inmutable ---");
    println!("❌ Si declaras: let capa = Capa {{ ... }}; y llamas capa.cambiar(10);");
    println!("   El compilador falla con error[E0596]: cannot borrow `capa` as mutable.");
    println!("✅ Solución: La instancia debe ser declarada como mutable: let mut capa = ...;");

    let mut capa = Capa { neuronas: 64 };
    capa.cambiar(128);
    println!("   Demostración corregida: neuronas = {}\n", capa.neuronas);

    println!("========================================================");
    println!("🎉 ¡Laboratorio de errores superado!");
    println!("========================================================");
}
