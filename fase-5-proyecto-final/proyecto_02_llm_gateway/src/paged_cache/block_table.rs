//! Administrador de Bloques Físicos y Tablas de Paginación

use std::collections::VecDeque;

/// Administrador y asignador de bloques físicos de memoria en el pool de KV-Cache
#[derive(Debug, Clone)]
pub struct BlockAllocator {
    pub block_size: usize,
    pub total_blocks: usize,
    free_blocks: VecDeque<usize>,
}

impl BlockAllocator {
    pub fn new(total_blocks: usize, block_size: usize) -> Self {
        let mut free_blocks = VecDeque::with_capacity(total_blocks);
        for id in 0..total_blocks {
            free_blocks.push_back(id);
        }
        Self {
            block_size,
            total_blocks,
            free_blocks,
        }
    }

    /// Asigna un nuevo bloque físico libre si está disponible
    pub fn allocate(&mut self) -> Option<usize> {
        self.free_blocks.pop_front()
    }

    /// Asigna múltiples bloques necesarios para albergar una longitud de tokens dada
    pub fn allocate_blocks_for_length(&mut self, num_tokens: usize) -> Option<Vec<usize>> {
        let blocks_needed = (num_tokens + self.block_size - 1) / self.block_size;
        if self.free_blocks.len() < blocks_needed {
            return None; // No hay suficiente memoria contigua disponible
        }

        let mut allocated = Vec::with_capacity(blocks_needed);
        for _ in 0..blocks_needed {
            allocated.push(self.free_blocks.pop_front().unwrap());
        }
        Some(allocated)
    }

    /// Libera un bloque físico de vuelta al pool
    pub fn free(&mut self, block_id: usize) {
        if block_id < self.total_blocks && !self.free_blocks.contains(&block_id) {
            self.free_blocks.push_back(block_id);
        }
    }

    /// Libera una lista de bloques físicos de una secuencia finalizada
    pub fn free_all(&mut self, block_ids: &[usize]) {
        for &id in block_ids {
            self.free(id);
        }
    }

    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_used_blocks(&self) -> usize {
        self.total_blocks - self.free_blocks.len()
    }

    pub fn memory_utilization(&self) -> f32 {
        if self.total_blocks == 0 {
            0.0
        } else {
            (self.num_used_blocks() as f32) / (self.total_blocks as f32)
        }
    }
}
