//! Índice Vectorial Denso con Cálculo de Similitud Coseno Acelerado

#[derive(Debug, Default, Clone)]
pub struct DenseIndex {
    pub vectors: Vec<(String, Vec<f32>)>,
    pub dim: usize,
}

impl DenseIndex {
    pub fn new(dim: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dim,
        }
    }

    pub fn insert(&mut self, chunk_id: String, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dim,
            "Dimensión del embedding incompatible con el índice"
        );
        self.vectors.push((chunk_id, embedding));
    }

    /// Producto punto optimizado con desenrollado de bucle (auto-vectorizable por LLVM / AVX2)
    #[inline(always)]
    fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len();
        let chunks = len / 8;
        let remainder = len % 8;

        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;
        let mut acc4 = 0.0f32;
        let mut acc5 = 0.0f32;
        let mut acc6 = 0.0f32;
        let mut acc7 = 0.0f32;

        for i in 0..chunks {
            let offset = i * 8;
            acc0 += a[offset] * b[offset];
            acc1 += a[offset + 1] * b[offset + 1];
            acc2 += a[offset + 2] * b[offset + 2];
            acc3 += a[offset + 3] * b[offset + 3];
            acc4 += a[offset + 4] * b[offset + 4];
            acc5 += a[offset + 5] * b[offset + 5];
            acc6 += a[offset + 6] * b[offset + 6];
            acc7 += a[offset + 7] * b[offset + 7];
        }

        let mut total = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);

        let rem_start = chunks * 8;
        for i in 0..remainder {
            total += a[rem_start + i] * b[rem_start + i];
        }

        total
    }

    /// Búsqueda de los Top-K fragmentos más cercanos por similitud coseno
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Vec<(String, f32)> {
        assert_eq!(query_vector.len(), self.dim);

        let mut scored: Vec<(String, f32)> = self
            .vectors
            .iter()
            .map(|(id, vec)| {
                let score = Self::dot_product(query_vector, vec);
                (id.clone(), score)
            })
            .collect();

        // Ordenar descendentemente por score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }
}
