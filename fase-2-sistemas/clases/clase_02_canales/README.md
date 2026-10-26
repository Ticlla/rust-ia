# 📦 Clase 02: Canales de Comunicación (`mpsc`) y DataLoaders en IA

Bienvenido a la segunda clase de la **Fase 2**. En esta unidad aprenderás a conectar hilos entre sí utilizando **canales de paso de mensajes** (*Message Passing*), el patrón fundamental para construir cargadores de datos eficientes (*DataLoaders*) que alimenten tus modelos de IA sin cuellos de botella.

---

## 📑 Contenido de la Clase

1. [La Filosofía del Paso de Mensajes](#1-la-filosofía-del-paso-de-mensajes)
2. [Anatomía de `mpsc::channel()`](#2-anatomía-de-mpscchannel)
3. [Transferencia de Ownership a través del Canal](#3-transferencia-de-ownership-a-través-del-canal)
4. [Múltiples Productores (`tx.clone()`) y el Cierre del Canal](#4-múltiples-productores-txclone-y-el-cierre-del-canal)
5. [Ejecutables de la Clase](#5-ejecutables-de-la-clase)

---

## 1. La Filosofía del Paso de Mensajes

Existe un proverbio célebre en la programación de sistemas concurrentes:  
> *"No te comuniques compartiendo memoria; comparte memoria comunicándote."*

En lugar de que 4 hilos intenten escribir en el mismo array al mismo tiempo (peleando por cerrojos y arriesgando bloqueos), **creamos un canal (cinta transportadora)**:
* Los hilos productores leen del disco y depositan los datos en el canal.
* El hilo consumidor (la GPU o el entrenador) recoge los datos y los procesa de forma continua.

---

## 2. Anatomía de `mpsc::channel()`

`mpsc` significa **Multiple Producer, Single Consumer** (Múltiples Productores, Un Solo Consumidor):

```rust
use std::sync::mpsc;

// tx = Transmitter (envía)
// rx = Receiver (recibe)
let (tx, rx) = mpsc::channel();
```

---

## 3. Transferencia de Ownership a través del Canal

Cuando haces `tx.send(tensor)`:
* `send()` **toma posesión** del tensor.
* El tensor viaja físicamente por el canal hacia el hilo receptor.
* En el hilo emisor, la variable original **queda destruida e invalidada**.
* **Ventaja**: Es imposible que dos hilos lean o modifiquen el mismo tensor al mismo tiempo. Cero *Data Races*.

---

## 4. Múltiples Productores (`tx.clone()`) y el Cierre del Canal

Puedes clonar `tx` tantas veces como hilos trabajadores tengas:
```rust
let tx1 = tx.clone();
let tx2 = tx.clone();
```

> [!WARNING]
> El receptor `rx` solo sabe que la transmisión ha terminado cuando **TODOS los clones de `tx` han muerto**.  
> Por eso, debes hacer `drop(tx)` sobre la variable original de `main` antes de empezar a consumir con `for lote in rx`.

---

## 5. Ejecutables de la Clase

```bash
# 1. Probar la teoría y el DataLoader con 2 workers:
cargo run --bin c02_teoria

# 2. Explorar el laboratorio de errores comunes:
cargo run --bin c02_errores

# 3. Correr el reto de normalización en paralelo:
cargo run --bin c02_ejercicios

# 4. Validar las pruebas unitarias:
cargo test --bin c02_ejercicios
```
