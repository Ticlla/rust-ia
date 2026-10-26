# Clase 05: Vectorización y SIMD Explícito (AVX2 / AVX-512) ⚡🏎️

En Inteligencia Artificial, el $90\%$ del tiempo de cómputo en CPU se gasta en **sumas y multiplicaciones de vectores y matrices** (productos punto para calcular similitud coseno, capas densas y proyecciones de atención).

Por defecto, una CPU ejecuta instrucciones **Escalares**: toma un número flotante de 32 bits (`f32`), lo multiplica por otro y produce un resultado por ciclo de reloj.

Con **SIMD (*Single Instruction, Multiple Data*)**, la CPU utiliza registros vectoriales gigantes (**YMM** de 256 bits y **ZMM** de 512 bits) para procesar **8 o 16 floats simultáneamente en un solo ciclo de reloj**.

---

## 🎯 Objetivos de la Clase
1. Comprender la diferencia arquitectónica entre **Cómputo Escalar vs Vectorial (SIMD)**.
2. Explorar los registros físicos de tu CPU Intel Core i5:
   - **AVX2:** Registros de 256 bits (procesa $8 \times \text{f32}$ a la vez).
   - **FMA (*Fused Multiply-Add*):** Calcula $a \times b + c$ en un solo ciclo sin errores de redondeo intermedios.
   - **AVX-512 y VNNI:** Instrucciones vectoriales para redes neuronales profundas.
3. Escribir **Intrínsecos x86_64 Explícitos** en Rust (`_mm256_loadu_ps`, `_mm256_fmadd_ps`).
4. Implementar el **Manejo Seguro del Residuo** (*Remainder Handling*) para vectores de cualquier dimensión.

---

## 🏗️ Comparativa: Escalar vs SIMD en Hardware

| Característica | Modo Escalar Tradicional | Modo Vectorial SIMD (AVX2 + FMA) |
|---|---|---|
| **Floats por Ciclo** | 1 float (`f32`) | **8 floats (`f32`) a la vez** |
| **Registros de CPU** | Registros estándar (32/64 bits) | **Registros YMM (256 bits)** |
| **Operación FMA** | Multiplica, guarda, luego suma (2 ciclos) | **Multiplica y suma en 1 solo ciclo** |
| **Rendimiento Real**| Base ($1\times$) | **$4\times$ a $8\times$ más rápido** |

---

## 🚀 Comandos de Ejecución

```bash
# 1. Demostración teórica y benchmark en vivo de aceleración:
cargo run --release --bin c05_teoria

# 2. Laboratorio de errores (desalineación de memoria y olvido del residuo):
cargo run --bin c05_errores

# 3. Ejecutar suite de pruebas unitarias:
cargo test --bin c05_ejercicios
```
