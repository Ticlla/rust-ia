# 📚 Clase 06: Borrowing (Préstamos) y Referencias (`&` vs `&mut`)

En esta clase aprenderás cómo pedir datos prestados en Rust sin perder la propiedad y sin gastar ciclos de CPU copiando buffers en memoria. Entenderás la diferencia entre referencias de solo lectura (`&T`) y referencias exclusivas de modificación (`&mut T`), y cómo el compilador erradica las condiciones de carrera (*Data Races*).

---

## 📑 Contenido
1. [Referencias Inmutables `&T` (Lectura compartida)](#1-referencias-inmutables)
2. [Referencias Mutables `&mut T` (Escritura exclusiva)](#2-referencias-mutables)
3. [La Regla de Oro del Borrowing](#3-la-regla-de-oro)
4. [Comandos para Ejecutar](#4-comandos-para-ejecutar)

---

## 4. Comandos para Ejecutar
```bash
cargo run --bin c06_teoria
cargo run --bin c06_errores
cargo run --bin c06_ejercicios
```
