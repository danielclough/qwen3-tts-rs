use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use crate::audio::{
    decoder::{config::TokenizerV2DecoderConfig, residual_unit::DecoderBlock},
    quantizer::SplitResidualVectorQuantizer,
    tokenizer::{
        snake_beta::SnakeBeta,
        v2::{
            causal_conv::{CausalConv1d, CausalConvTranspose1d},
            convnext::ConvNeXtBlock,
            transformer::TokenizerV2DecoderTransformer,
        },
    },
};

/// Complete decoder for the 12Hz tokenizer.
#[derive(Debug)]
pub struct TokenizerV2Decoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: TokenizerV2DecoderTransformer,
    upsample_blocks: Vec<(CausalConvTranspose1d, ConvNeXtBlock)>,
    decoder_pre_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    decoder_final_act: SnakeBeta,
    decoder_final_conv: CausalConv1d,
    total_upsample: usize,
}

impl TokenizerV2Decoder {
    pub fn new(
        config: &TokenizerV2DecoderConfig,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        tracing::debug!(
            codebook_dim = config.codebook_dim,
            codebook_size = config.codebook_size,
            hidden_size = config.hidden_size,
            latent_dim = config.latent_dim,
            intermediate_size = config.intermediate_size,
            num_quantizers = config.num_quantizers,
            num_hidden_layers = config.num_hidden_layers,
            decoder_dim = config.decoder_dim,
            upsample_rates = ?config.upsample_rates,
            upsampling_ratios = ?config.upsampling_ratios,
            sliding_window = config.sliding_window,
            num_attention_heads = config.num_attention_heads,
            num_kv_heads = config.num_key_value_heads,
            "decoder config"
        );

        // Vector quantizer
        let quantizer = SplitResidualVectorQuantizer::new(
            config.num_quantizers,
            1, // n_q_semantic
            config.codebook_dim / 2,
            Some(config.codebook_dim),
            Some(config.codebook_dim),
            config.codebook_size,
            vb.pp("quantizer"),
        )?;

        // Pre-conv
        let pre_conv = CausalConv1d::new(
            config.codebook_dim,
            config.latent_dim,
            3,
            1,
            1,
            1,
            vb.pp("pre_conv"),
        )?;

        // Transformer
        let pre_transformer =
            TokenizerV2DecoderTransformer::new(config, use_flash_attn, vb.pp("pre_transformer"))?;

        // Upsampling blocks with ConvNeXt
        let upsample_blocks = config
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(i, &factor)| {
                let upsample = CausalConvTranspose1d::new(
                    config.latent_dim,
                    config.latent_dim,
                    factor,
                    factor,
                    vb.pp(format!("upsample.{}.0", i)),
                )?;
                let convnext =
                    ConvNeXtBlock::new(config.latent_dim, vb.pp(format!("upsample.{}.1", i)))?;
                Ok((upsample, convnext))
            })
            .collect::<Result<Vec<_>>>()?;

        // Decoder vocoder
        let decoder_pre_conv = CausalConv1d::new(
            config.latent_dim,
            config.decoder_dim,
            7,
            1,
            1,
            1,
            vb.pp("decoder.0"),
        )?;

        let mut decoder_blocks = Vec::new();
        let mut current_dim = config.decoder_dim;
        for (i, &rate) in config.upsample_rates.iter().enumerate() {
            let next_dim = current_dim / 2;
            decoder_blocks.push(DecoderBlock::new(
                current_dim,
                next_dim,
                rate,
                vb.pp(format!("decoder.{}", i + 1)),
            )?);
            current_dim = next_dim;
        }

        let output_dim = config.decoder_dim / (1 << config.upsample_rates.len());
        let decoder_final_act = SnakeBeta::new(
            output_dim,
            vb.pp(format!("decoder.{}", config.upsample_rates.len() + 1)),
        )?;
        let decoder_final_conv = CausalConv1d::new(
            output_dim,
            1,
            7,
            1,
            1,
            1,
            vb.pp(format!("decoder.{}", config.upsample_rates.len() + 2)),
        )?;

        let total_upsample = config.total_upsample();

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample_blocks,
            decoder_pre_conv,
            decoder_blocks,
            decoder_final_act,
            decoder_final_conv,
            total_upsample,
        })
    }

    pub fn load(
        config: &TokenizerV2DecoderConfig,
        use_flash_attn: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        Self::new(config, use_flash_attn, vb)
    }

    /// Decode codes to waveform.
    ///
    /// Input: (batch, num_quantizers, seq_len) code indices
    /// Output: (batch, samples) audio waveform
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        fn debug_tensor(name: &str, t: &Tensor) {
            if !tracing::enabled!(tracing::Level::DEBUG) {
                return;
            }
            let shape = t.dims();
            // Convert to F32 first for statistics
            if let Ok(t_f32) = t.to_dtype(candle_core::DType::F32)
                && let (Ok(min), Ok(max), Ok(mean)) = (
                    t_f32
                        .flatten_all()
                        .and_then(|f| f.min(0))
                        .and_then(|m| m.to_scalar::<f32>()),
                    t_f32
                        .flatten_all()
                        .and_then(|f| f.max(0))
                        .and_then(|m| m.to_scalar::<f32>()),
                    t_f32
                        .flatten_all()
                        .and_then(|f| f.mean_all())
                        .and_then(|m| m.to_scalar::<f32>()),
                )
            {
                tracing::debug!(
                    stage = name,
                    shape = ?shape,
                    min = format!("{:.4}", min),
                    max = format!("{:.4}", max),
                    mean = format!("{:.6}", mean),
                    "decoder tensor"
                );
                return;
            }
            tracing::debug!(stage = name, shape = ?shape, "decoder tensor");
        }

        tracing::debug!(shape = ?codes.dims(), "decoder input codes");

        // Dequantize
        let hidden = self.quantizer.decode(codes)?;
        debug_tensor("after quantizer", &hidden);

        // Pre-conv: (batch, codebook_dim, seq) -> (batch, latent_dim, seq)
        let hidden = self.pre_conv.forward(&hidden)?;
        debug_tensor("after pre_conv", &hidden);

        // Transpose for transformer: (batch, latent_dim, seq) -> (batch, seq, latent_dim)
        let hidden = hidden.transpose(1, 2)?;

        // Transformer
        let hidden = self.pre_transformer.forward(&hidden)?;
        debug_tensor("after transformer", &hidden);

        // Transpose back: (batch, seq, latent_dim) -> (batch, latent_dim, seq)
        let mut hidden = hidden.transpose(1, 2)?;

        // Upsampling with ConvNeXt
        for (i, (upsample, convnext)) in self.upsample_blocks.iter().enumerate() {
            hidden = upsample.forward(&hidden)?;
            hidden = convnext.forward(&hidden)?;
            debug_tensor(&format!("after upsample_block[{}]", i), &hidden);
        }

        // Vocoder
        hidden = self.decoder_pre_conv.forward(&hidden)?;
        debug_tensor("after decoder_pre_conv", &hidden);

        for (i, block) in self.decoder_blocks.iter().enumerate() {
            hidden = block.forward(&hidden)?;
            debug_tensor(&format!("after decoder_block[{}]", i), &hidden);
        }

        hidden = self.decoder_final_act.forward(&hidden)?;
        debug_tensor("after final_act", &hidden);

        let wav = self.decoder_final_conv.forward(&hidden)?;
        debug_tensor("after final_conv", &wav);

        // Clamp output
        let wav = wav.clamp(-1.0, 1.0)?;
        debug_tensor("after clamp", &wav);

        // Squeeze channel dimension: (batch, 1, samples) -> (batch, samples)
        wav.squeeze(1)
    }

    /// Decode with chunking for long sequences.
    pub fn chunked_decode(
        &self,
        codes: &Tensor,
        chunk_size: usize,
        left_context_size: usize,
    ) -> Result<Tensor> {
        let seq_len = codes.dim(2)?;
        let mut wavs = Vec::new();
        let mut start_index = 0;

        while start_index < seq_len {
            let end_index = (start_index + chunk_size).min(seq_len);
            let context_size = if start_index > left_context_size {
                left_context_size
            } else {
                start_index
            };

            let codes_chunk = codes.narrow(
                2,
                start_index - context_size,
                end_index - start_index + context_size,
            )?;
            let wav_chunk = self.forward(&codes_chunk)?;

            // Trim context from output
            let context_samples = context_size * self.total_upsample;
            let wav_chunk =
                wav_chunk.narrow(1, context_samples, wav_chunk.dim(1)? - context_samples)?;

            wavs.push(wav_chunk);
            start_index = end_index;
        }

        // Concatenate all chunks
        Tensor::cat(&wavs.iter().collect::<Vec<_>>(), 1)
    }
}
