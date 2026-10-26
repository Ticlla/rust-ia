//! Gestor de Almacenamiento Tensorial Paginado en Memoria con Candle

use super::block_table::BlockAllocator;
use candle_core::{Device, Result, Tensor};
use std::collections::HashMap;

/// Almacena y mapea los bloques físicos de KV-Cache en memoria RAM/VRAM
pub struct PagedKvCacheManager {
    pub allocator: BlockAllocator,
    pub block_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub dev: Device,
    /// Mapa de capa -> physical_block_id -> (K_block, V_block)
    /// Cada bloque tiene forma [block_size, n_heads, head_dim]
    pub storage: Vec<HashMap<usize, (Tensor, Tensor)>>,
}

impl PagedKvCacheManager {
    pub fn new(
        total_blocks: usize,
        block_size: usize,
        n_layers: usize,
        n_heads: usize,
        head_dim: usize,
        dev: &Device,
    ) -> Self {
        let allocator = BlockAllocator::new(total_blocks, block_size);
        let mut storage = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            storage.push(HashMap::new());
        }

        Self {
            allocator,
            block_size,
            n_layers,
            n_heads,
            head_dim,
            dev: dev.clone(),
            storage,
        }
    }

    /// Guarda un vector K y V para un token en la posición física asignada
    pub fn write_token(
        &mut self,
        layer_idx: usize,
        physical_block_id: usize,
        offset_in_block: usize,
        k_token: &Tensor, // [n_heads, head_dim]
        v_token: &Tensor, // [n_heads, head_dim]
    ) -> Result<()> {
        let layer_storage = &mut self.storage[layer_idx];

        if !layer_storage.contains_key(&physical_block_id) {
            // Inicializar bloque vacío con ceros
            let empty_k = Tensor::zeros(
                (self.block_size, self.n_heads, self.head_dim),
                candle_core::DType::F32,
                &self.dev,
            )?;
            let empty_v = Tensor::zeros(
                (self.block_size, self.n_heads, self.head_dim),
                candle_core::DType::F32,
                &self.dev,
            )?;
            layer_storage.insert(physical_block_id, (empty_k, empty_v));
        }

        let (k_block, v_block) = layer_storage.get_mut(&physical_block_id).unwrap();

        // Slice-assign: Reemplazamos en el offset_in_block usando concatenación pedagógica de slices
        let k_t = k_token.unsqueeze(0)?; // [1, n_heads, head_dim]
        let v_t = v_token.unsqueeze(0)?;

        let mut k_parts = Vec::new();
        let mut v_parts = Vec::new();

        if offset_in_block > 0 {
            k_parts.push(k_block.narrow(0, 0, offset_in_block)?);
            v_parts.push(v_block.narrow(0, 0, offset_in_block)?);
        }
        k_parts.push(k_t);
        v_parts.push(v_t);

        if offset_in_block + 1 < self.block_size {
            let rem = self.block_size - (offset_in_block + 1);
            k_parts.push(k_block.narrow(0, offset_in_block + 1, rem)?);
            v_parts.push(v_block.narrow(0, offset_in_block + 1, rem)?);
        }

        let k_refs: Vec<&Tensor> = k_parts.iter().collect();
        let v_refs: Vec<&Tensor> = v_parts.iter().collect();

        *k_block = Tensor::cat(&k_refs, 0)?;
        *v_block = Tensor::cat(&v_refs, 0)?;

        Ok(())
    }

    /// Reconstruye el contexto completo K y V para una secuencia a partir de su Block Table
    pub fn gather_sequence_kv(
        &self,
        layer_idx: usize,
        block_table: &[usize],
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if seq_len == 0 || block_table.is_empty() {
            let empty = Tensor::zeros(
                (1, self.n_heads, 0, self.head_dim),
                candle_core::DType::F32,
                &self.dev,
            )?;
            return Ok((empty.clone(), empty));
        }

        let layer_storage = &self.storage[layer_idx];
        let mut k_chunks = Vec::new();
        let mut v_chunks = Vec::new();
        let mut tokens_gathered = 0;

        for &block_id in block_table {
            if let Some((k_block, v_block)) = layer_storage.get(&block_id) {
                let tokens_in_this_block = (seq_len - tokens_gathered).min(self.block_size);
                k_chunks.push(k_block.narrow(0, 0, tokens_in_this_block)?);
                v_chunks.push(v_block.narrow(0, 0, tokens_in_this_block)?);
                tokens_gathered += tokens_in_this_block;
                if tokens_gathered >= seq_len {
                    break;
                }
            }
        }

        let k_refs: Vec<&Tensor> = k_chunks.iter().collect();
        let v_refs: Vec<&Tensor> = v_chunks.iter().collect();

        // Concatenar a lo largo del eje temporal (dim 0) -> [seq_len, n_heads, head_dim]
        let full_k = Tensor::cat(&k_refs, 0)?;
        let full_v = Tensor::cat(&v_refs, 0)?;

        // Transpose a formato estándar [1, n_heads, seq_len, head_dim]
        let full_k = full_k.transpose(0, 1)?.unsqueeze(0)?;
        let full_v = full_v.transpose(0, 1)?.unsqueeze(0)?;

        Ok((full_k, full_v))
    }

    /// Libera los bloques asociados a una secuencia finalizada
    pub fn free_sequence(&mut self, block_table: &[usize]) {
        for &block_id in block_table {
            for layer in 0..self.n_layers {
                self.storage[layer].remove(&block_id);
            }
        }
        self.allocator.free_all(block_table);
    }
}
