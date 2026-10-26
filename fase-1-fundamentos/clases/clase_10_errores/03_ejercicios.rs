//! 🦀 CLASE 10: RETOS Y EJERCICIOS PRÁCTICOS EN IA
//! Ejecutar con: cargo run --bin c10_ejercicios

fn parsear_hiperparametro(raw: &str) -> Result<f32, String> {
    raw.trim().parse::<f32>().map_err(|e| format!("Formato inválido: {}", e))
}

fn main() {
    println!("========================================================");
    println!("🎯 CLASE 10 [EJERCICIOS PRÁCTICOS]: Option y Result");
    println!("========================================================\n");

    let res_ok = parsear_hiperparametro(" 0.001 ");
    let res_err = parsear_hiperparametro("invalido");

    assert!(res_ok.is_ok());
    assert!(res_err.is_err());

    println!("Parseo exitoso : {:?}", res_ok);
    println!("Error capturado: {:?}\n", res_err);

    println!("========================================================");
    println!("🎉 ¡TODOS LOS RETOS DE LA CLASE 10 FUERON SUPERADOS!");
    println!("========================================================");
}
