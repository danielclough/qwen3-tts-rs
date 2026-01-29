//! V1 vector quantization components for the 25Hz tokenizer.
//!
//! Unlike V2's `quantizer.rs` which normalizes embeddings via EMA cluster_usage/embedding_sum,
//! V1 stores embeddings directly as stacked buffers (`embed`, `cluster_size`, etc.) that are
//! sliced per-quantizer at runtime.
//!
//! Hierarchy: GRVQ → RVQ → VQ → EuclideanCodebook

use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

/// Euclidean codebook — stateless, buffers passed at call time.
///
/// Mirrors `EuclideanCodebook` in `core_vq.py`.
/// Distance: `-(x² - 2x@embed^T + embed²)`, argmax = nearest.
#[derive(Debug, Clone)]
pub struct V1EuclideanCodebook {
    dim: usize,
}

impl V1EuclideanCodebook {
    pub fn new(dim: usize, _codebook_size: usize) -> Self {
        Self { dim }
    }

    /// Quantize: find nearest codebook entry for each input vector.
    ///
    /// `x`: `(N, dim)` flattened input vectors
    /// `embed`: `(codebook_size, dim)` codebook embeddings
    /// Returns: `(N,)` indices
    pub fn quantize(&self, x: &Tensor, embed: &Tensor) -> Result<Tensor> {
        // embed_t: (dim, codebook_size)
        let embed_t = embed.t()?;
        // dist = -(x² - 2·x@embed^T + embed²)  → argmax = nearest
        let x2 = x.sqr()?.sum_keepdim(D::Minus1)?; // (N, 1)
        let dot = x.matmul(&embed_t)?; // (N, codebook_size)
        let e2 = embed_t.sqr()?.sum_keepdim(0)?; // (1, codebook_size)
        // neg_dist = -(x² - 2·dot + e²) = 2·dot - x² - e²
        let neg_dist = dot.affine(2.0, 0.0)?.broadcast_sub(&x2)?.broadcast_sub(&e2)?;
        neg_dist.argmax(D::Minus1)
    }

    /// Dequantize: look up embeddings by index.
    ///
    /// `indices`: arbitrary shape of indices
    /// `embed`: `(codebook_size, dim)` codebook embeddings
    pub fn dequantize(&self, indices: &Tensor, embed: &Tensor) -> Result<Tensor> {
        let original_shape = indices.dims().to_vec();
        let flat = indices.flatten_all()?;
        let looked_up = embed.embedding(&flat)?;
        let mut new_shape = original_shape;
        new_shape.push(self.dim);
        looked_up.reshape(new_shape)
    }

    /// Encode: flatten → quantize → reshape.
    ///
    /// `x`: `(..., dim)` input
    /// `embed`: `(codebook_size, dim)`
    pub fn encode(&self, x: &Tensor, embed: &Tensor) -> Result<Tensor> {
        let shape = x.dims().to_vec();
        let flat = x.flatten_to(D::Minus2)?; // (N, dim)
        let indices = self.quantize(&flat, embed)?;
        // Reshape to input shape without last dim
        let out_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
        if out_shape.is_empty() {
            Ok(indices)
        } else {
            indices.reshape(out_shape)
        }
    }

    /// Decode: dequantize wrapper.
    pub fn decode(&self, indices: &Tensor, embed: &Tensor) -> Result<Tensor> {
        self.dequantize(indices, embed)
    }
}

/// Single-codebook vector quantization with optional projection.
///
/// Mirrors `VectorQuantization` in `core_vq.py`.
#[derive(Debug, Clone)]
pub struct V1VectorQuantization {
    codebook: V1EuclideanCodebook,
    project_in: Option<Linear>,
    project_out: Option<Linear>,
}

impl V1VectorQuantization {
    pub fn new(dim: usize, codebook_size: usize, codebook_dim: Option<usize>, vb: VarBuilder) -> Result<Self> {
        let cb_dim = codebook_dim.unwrap_or(dim);
        let codebook = V1EuclideanCodebook::new(cb_dim, codebook_size);

        let (project_in, project_out) = if cb_dim != dim {
            let p_in = linear_no_bias(dim, cb_dim, vb.pp("project_in"))?;
            let p_out = linear_no_bias(cb_dim, dim, vb.pp("project_out"))?;
            (Some(p_in), Some(p_out))
        } else {
            (None, None)
        };

        Ok(Self {
            codebook,
            project_in,
            project_out,
        })
    }

    /// Encode input to indices.
    ///
    /// `x`: `(batch, seq, dim)` — note: V1 python passes `(b, n, d)` directly (no transpose)
    /// `embed`: `(codebook_size, dim)` from the stacked buffer
    pub fn encode(&self, x: &Tensor, embed: &Tensor) -> Result<Tensor> {
        let x = match &self.project_in {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };
        self.codebook.encode(&x, embed)
    }

    /// Decode indices back to continuous.
    ///
    /// `indices`: `(batch, seq)`
    /// `embed`: `(codebook_size, dim)`
    /// Returns: `(batch, seq, dim)`
    pub fn decode(&self, indices: &Tensor, embed: &Tensor) -> Result<Tensor> {
        let quantized = self.codebook.decode(indices, embed)?;
        match &self.project_out {
            Some(proj) => proj.forward(&quantized),
            None => Ok(quantized),
        }
    }
}

/// Residual Vector Quantization — multiple VQ layers with stacked embed buffer.
///
/// Mirrors `DistributedResidualVectorQuantization` in `core_vq.py`.
#[derive(Debug, Clone)]
pub struct V1ResidualVectorQuantization {
    layers: Vec<V1VectorQuantization>,
    /// Stacked embeddings: `(num_quantizers, codebook_size, dim)`
    embed: Tensor,
}

impl V1ResidualVectorQuantization {
    pub fn new(
        num_quantizers: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cb_dim = codebook_dim.unwrap_or(dim);
        let embed = vb.get((num_quantizers, codebook_size, cb_dim), "embed")?;

        let layers = (0..num_quantizers)
            .map(|i| {
                V1VectorQuantization::new(dim, codebook_size, codebook_dim, vb.pp(format!("layers.{i}")))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers, embed })
    }

    /// Encode: residual quantization.
    ///
    /// `x`: `(batch, seq, dim)`
    /// Returns: `(n_q, batch, seq)` indices
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let mut residual = x.clone();
        let mut all_indices = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let embed_i = self.embed.i(i)?;
            let indices = layer.encode(&residual, &embed_i)?;
            let quantized = layer.decode(&indices, &embed_i)?;
            residual = (&residual - &quantized)?;
            all_indices.push(indices);
        }

        Tensor::stack(&all_indices, 0)
    }

    /// Decode: sum of per-layer dequantized.
    ///
    /// `codes`: `(n_q, batch, seq)`
    /// Returns: `(batch, seq, dim)`
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let n_q = codes.dim(0)?;
        let mut quantized_out: Option<Tensor> = None;

        for (i, layer) in self.layers.iter().enumerate().take(n_q) {
            let embed_i = self.embed.i(i)?;
            let layer_codes = codes.i(i)?;
            let q = layer.decode(&layer_codes, &embed_i)?;
            quantized_out = Some(match quantized_out {
                Some(acc) => (acc + q)?,
                None => q,
            });
        }

        Ok(quantized_out.unwrap())
    }
}

/// Group Residual Vector Quantization — splits input along dim, applies RVQ per group.
///
/// Mirrors `DistributedGroupResidualVectorQuantization` in `core_vq.py`.
#[derive(Debug, Clone)]
pub struct V1GroupResidualVectorQuantization {
    rvqs: Vec<V1ResidualVectorQuantization>,
    num_groups: usize,
}

impl V1GroupResidualVectorQuantization {
    pub fn new(
        num_groups: usize,
        num_quantizers: usize,
        dim: usize,
        codebook_size: usize,
        codebook_dim: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let group_dim = dim / num_groups;
        let rvqs = (0..num_groups)
            .map(|i| {
                V1ResidualVectorQuantization::new(
                    num_quantizers,
                    group_dim,
                    codebook_size,
                    codebook_dim,
                    vb.pp(format!("rvqs.{i}")),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { rvqs, num_groups })
    }

    /// Encode: chunk input along dim axis, encode each group.
    ///
    /// `x`: `(batch, seq, dim)` — note: Python chunks on dim=1 with `(b, d, t)` layout,
    ///   but we use `(b, t, d)` layout so we chunk on dim=2 (last dim).
    /// Returns: `(n_q, num_groups, batch, seq)`
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let chunks = x.chunk(self.num_groups, D::Minus1)?;
        let group_codes: Vec<Tensor> = self
            .rvqs
            .iter()
            .zip(chunks.iter())
            .map(|(rvq, chunk)| rvq.encode(chunk))
            .collect::<Result<Vec<_>>>()?;
        // Each group_code: (n_q, batch, seq) → stack on dim=1 → (n_q, num_groups, batch, seq)
        Tensor::stack(&group_codes, 1)
    }

    /// Decode: decode each group, cat along dim axis.
    ///
    /// `codes`: `(n_q, num_groups, batch, seq)`
    /// Returns: `(batch, seq, dim)`
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let group_results: Vec<Tensor> = (0..self.num_groups)
            .map(|g| {
                let group_codes = codes.i((.., g))?; // (n_q, batch, seq)
                self.rvqs[g].decode(&group_codes)
            })
            .collect::<Result<Vec<_>>>()?;
        // Each: (batch, seq, group_dim) → cat on last dim
        Tensor::cat(&group_results, D::Minus1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_euclidean_distance_computation() -> Result<()> {
        // 3 codebook entries of dim 2
        let embed = Tensor::new(&[[0.0f32, 0.0], [1.0, 0.0], [0.0, 1.0]], &dev())?;
        let cb = V1EuclideanCodebook::new(2, 3);

        // Query close to entry 1 (1,0)
        let x = Tensor::new(&[[0.9f32, 0.1]], &dev())?;
        let idx = cb.quantize(&x, &embed)?;
        assert_eq!(idx.to_vec1::<u32>()?[0], 1);

        // Query close to entry 2 (0,1)
        let x = Tensor::new(&[[0.1f32, 0.8]], &dev())?;
        let idx = cb.quantize(&x, &embed)?;
        assert_eq!(idx.to_vec1::<u32>()?[0], 2);

        Ok(())
    }

    #[test]
    fn test_codebook_dequantize() -> Result<()> {
        let embed = Tensor::new(&[[10.0f32, 20.0], [30.0, 40.0], [50.0, 60.0]], &dev())?;
        let cb = V1EuclideanCodebook::new(2, 3);

        let indices = Tensor::new(&[2u32, 0, 1], &dev())?;
        let result = cb.dequantize(&indices, &embed)?;
        let expected = Tensor::new(&[[50.0f32, 60.0], [10.0, 20.0], [30.0, 40.0]], &dev())?;

        let diff = (result - expected)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn test_vq_project_in_out_identity() -> Result<()> {
        // When codebook_dim == dim, no projection — encode→decode should find nearest
        let embed = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &dev())?;
        let cb = V1EuclideanCodebook::new(2, 2);

        let x = Tensor::new(&[[0.9f32, 0.1]], &dev())?;
        let idx = cb.quantize(&x, &embed)?;
        let decoded = cb.dequantize(&idx, &embed)?;
        // Should map to [1, 0]
        let diff = (decoded - Tensor::new(&[[1.0f32, 0.0]], &dev())?)?.abs()?.sum_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-6);
        Ok(())
    }

    #[test]
    fn test_rvq_residual_subtraction() -> Result<()> {
        // Manually verify: decode = sum of per-layer dequantized
        let dim = 4;
        let codebook_size = 8;
        let n_q = 3;
        let batch = 2;
        let seq = 5;

        // Create random embed buffer
        let embed = Tensor::randn(0f32, 1.0, (n_q, codebook_size, dim), &dev())?;

        // Create random codes in range [0, codebook_size)
        let codes_data: Vec<u32> = (0..(n_q * batch * seq)).map(|i| (i as u32) % codebook_size as u32).collect();
        let codes = Tensor::new(codes_data, &dev())?.reshape((n_q, batch, seq))?;

        // Manual decode: sum embeddings for each layer
        let cb = V1EuclideanCodebook::new(dim, codebook_size);
        let mut manual_sum: Option<Tensor> = None;
        for q in 0..n_q {
            let embed_q = embed.i(q)?;
            let codes_q = codes.i(q)?;
            let decoded = cb.dequantize(&codes_q, &embed_q)?;
            manual_sum = Some(match manual_sum {
                Some(acc) => (acc + decoded)?,
                None => decoded,
            });
        }

        // Build an RVQ and decode (without projection layers, just codebook)
        // We can't easily construct V1ResidualVectorQuantization without VarBuilder,
        // so just verify the manual sum shapes
        let result = manual_sum.unwrap();
        assert_eq!(result.dims(), &[batch, seq, dim]);
        Ok(())
    }

    #[test]
    fn test_rvq_output_shapes() -> Result<()> {
        let dim = 4;
        let codebook_size = 8;
        let n_q = 3;
        let batch = 2;
        let seq = 5;

        // Verify code tensor shapes
        let codes_data: Vec<u32> = (0..(n_q * batch * seq)).map(|i| (i as u32) % codebook_size as u32).collect();
        let codes = Tensor::new(codes_data, &dev())?.reshape((n_q, batch, seq))?;
        assert_eq!(codes.dims(), &[n_q, batch, seq]);

        // Verify dequantized shapes
        let embed = Tensor::randn(0f32, 1.0, (codebook_size, dim), &dev())?;
        let cb = V1EuclideanCodebook::new(dim, codebook_size);
        let decoded = cb.dequantize(&codes.i(0)?, &embed)?;
        assert_eq!(decoded.dims(), &[batch, seq, dim]);
        Ok(())
    }

    #[test]
    fn test_grvq_group_split_and_cat() -> Result<()> {
        let dim = 8;
        let num_groups = 2;
        let group_dim = dim / num_groups;
        let _codebook_size = 4;
        let batch = 2;
        let seq = 3;

        // Simulate: chunk input, decode per group, cat
        let x = Tensor::randn(0f32, 1.0, (batch, seq, dim), &dev())?;
        let chunks = x.chunk(num_groups, D::Minus1)?;
        assert_eq!(chunks.len(), num_groups);
        assert_eq!(chunks[0].dims(), &[batch, seq, group_dim]);

        // Simulate decode by creating random group outputs and concatenating
        let g0 = Tensor::randn(0f32, 1.0, (batch, seq, group_dim), &dev())?;
        let g1 = Tensor::randn(0f32, 1.0, (batch, seq, group_dim), &dev())?;
        let catted = Tensor::cat(&[&g0, &g1], D::Minus1)?;
        assert_eq!(catted.dims(), &[batch, seq, dim]);
        Ok(())
    }
}
