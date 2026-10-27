//! Microservicio con error intencional para evaluar grados de agenticidad.

pub fn calculate_tax(amount: f64) -> String {
    // ERROR E0308: expected `String`, found `f64`
    amount * 0.18
}

fn main() {
    let tax = calculate_tax(100.0);
    println!("Impuesto: {}", tax);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_tax() {
        assert_eq!(calculate_tax(100.0), "18.00");
    }
}
