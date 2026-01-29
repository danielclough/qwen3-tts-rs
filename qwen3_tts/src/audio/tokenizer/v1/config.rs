use serde::Deserialize;

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

fn default_n_mels() -> usize { 128 }
fn default_n_ctx() -> usize { 1500 }
fn default_n_state() -> usize { 1280 }
fn default_n_head() -> usize { 20 }
fn default_n_layer() -> usize { 32 }
fn default_n_window() -> usize { 100 }
fn default_output_dim() -> usize { 3584 }
fn default_audio_vq_type() -> String { "GRVQ".to_string() }
fn default_audio_vq_layers() -> usize { 6 }
fn default_audio_vq_codebook_size() -> usize { 32768 }
fn default_audio_vq_codebook_dim() -> usize { 1280 }
fn default_audio_vq_pe() -> bool { true }
fn default_audio_vq_ds_rate() -> usize { 2 }

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV1EncoderConfig {
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
    #[serde(default = "default_n_ctx")]
    pub n_ctx: usize,
    #[serde(default = "default_n_state")]
    pub n_state: usize,
    #[serde(default = "default_n_head")]
    pub n_head: usize,
    #[serde(default = "default_n_layer")]
    pub n_layer: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_output_dim")]
    pub output_dim: usize,
    #[serde(default = "default_audio_vq_type")]
    pub audio_vq_type: String,
    #[serde(default = "default_audio_vq_layers")]
    pub audio_vq_layers: usize,
    #[serde(default = "default_audio_vq_codebook_size")]
    pub audio_vq_codebook_size: usize,
    #[serde(default = "default_audio_vq_codebook_dim")]
    pub audio_vq_codebook_dim: usize,
    #[serde(default = "default_audio_vq_pe")]
    pub audio_vq_pe: bool,
    #[serde(default = "default_audio_vq_ds_rate")]
    pub audio_vq_ds_rate: usize,
}

impl Default for TokenizerV1EncoderConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

// ---------------------------------------------------------------------------
// Decoder — DiT
// ---------------------------------------------------------------------------

fn default_hidden_size() -> usize { 1024 }
fn default_num_hidden_layers() -> usize { 22 }
fn default_num_attention_heads() -> usize { 16 }
fn default_ff_mult() -> usize { 2 }
fn default_emb_dim() -> usize { 512 }
fn default_head_dim() -> usize { 64 }
fn default_rope_theta() -> f64 { 10000.0 }
fn default_max_position_embeddings() -> usize { 32768 }
fn default_block_size() -> usize { 24 }
fn default_look_ahead_layers() -> Vec<usize> { vec![10] }
fn default_look_backward_layers() -> Vec<usize> { vec![0, 20] }
fn default_repeats() -> usize { 2 }
fn default_num_embeds() -> usize { 8193 }
fn default_dit_mel_dim() -> usize { 80 }
fn default_dropout() -> f64 { 0.1 }
fn default_enc_emb_dim() -> usize { 192 }
fn default_enc_dim() -> usize { 128 }
fn default_enc_channels() -> Vec<usize> { vec![256, 256, 256, 256, 768] }
fn default_enc_kernel_sizes() -> Vec<usize> { vec![5, 3, 3, 3, 1] }
fn default_enc_dilations() -> Vec<usize> { vec![1, 2, 3, 4, 1] }
fn default_enc_attention_channels() -> usize { 64 }
fn default_enc_res2net_scale() -> usize { 2 }
fn default_enc_se_channels() -> usize { 64 }

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV1DecoderDiTConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_ff_mult")]
    pub ff_mult: usize,
    #[serde(default = "default_emb_dim")]
    pub emb_dim: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    #[serde(default = "default_look_ahead_layers")]
    pub look_ahead_layers: Vec<usize>,
    #[serde(default = "default_look_backward_layers")]
    pub look_backward_layers: Vec<usize>,
    #[serde(default = "default_repeats")]
    pub repeats: usize,
    #[serde(default = "default_num_embeds")]
    pub num_embeds: usize,
    #[serde(default = "default_dit_mel_dim")]
    pub mel_dim: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default = "default_enc_emb_dim")]
    pub enc_emb_dim: usize,
    #[serde(default = "default_enc_dim")]
    pub enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
}

impl Default for TokenizerV1DecoderDiTConfig {
    fn default() -> Self {
        serde_json::from_str("{}").unwrap()
    }
}

// ---------------------------------------------------------------------------
// Decoder — BigVGAN
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV1DecoderBigVGANConfig {
    pub mel_dim: usize,
    pub upsample_initial_channel: usize,
    pub resblock_kernel_sizes: Vec<usize>,
    pub resblock_dilation_sizes: Vec<Vec<usize>>,
    pub upsample_rates: Vec<usize>,
    pub upsample_kernel_sizes: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Decoder (top-level)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV1DecoderConfig {
    pub dit_config: TokenizerV1DecoderDiTConfig,
    pub bigvgan_config: TokenizerV1DecoderBigVGANConfig,
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

fn default_input_sample_rate() -> usize { 24000 }
fn default_output_sample_rate() -> usize { 24000 }
fn default_decode_upsample_rate() -> usize { 1920 }
fn default_encode_downsample_rate() -> usize { 1920 }

#[derive(Debug, Clone, Deserialize)]
pub struct TokenizerV1Config {
    #[serde(default)]
    pub encoder_config: TokenizerV1EncoderConfig,
    pub decoder_config: TokenizerV1DecoderConfig,
    #[serde(default = "default_input_sample_rate")]
    pub input_sample_rate: usize,
    #[serde(default = "default_output_sample_rate")]
    pub output_sample_rate: usize,
    #[serde(default = "default_decode_upsample_rate")]
    pub decode_upsample_rate: usize,
    #[serde(default = "default_encode_downsample_rate")]
    pub encode_downsample_rate: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_defaults() {
        let cfg = TokenizerV1EncoderConfig::default();
        assert_eq!(cfg.n_mels, 128);
        assert_eq!(cfg.n_ctx, 1500);
        assert_eq!(cfg.n_state, 1280);
        assert_eq!(cfg.n_head, 20);
        assert_eq!(cfg.n_layer, 32);
        assert_eq!(cfg.n_window, 100);
        assert_eq!(cfg.output_dim, 3584);
        assert_eq!(cfg.audio_vq_type, "GRVQ");
        assert_eq!(cfg.audio_vq_layers, 6);
        assert_eq!(cfg.audio_vq_codebook_size, 32768);
        assert_eq!(cfg.audio_vq_codebook_dim, 1280);
        assert!(cfg.audio_vq_pe);
        assert_eq!(cfg.audio_vq_ds_rate, 2);
    }

    #[test]
    fn test_dit_config_defaults() {
        let cfg = TokenizerV1DecoderDiTConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 22);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.ff_mult, 2);
        assert_eq!(cfg.emb_dim, 512);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.rope_theta, 10000.0);
        assert_eq!(cfg.max_position_embeddings, 32768);
        assert_eq!(cfg.block_size, 24);
        assert_eq!(cfg.look_ahead_layers, vec![10]);
        assert_eq!(cfg.look_backward_layers, vec![0, 20]);
        assert_eq!(cfg.repeats, 2);
        assert_eq!(cfg.num_embeds, 8193);
        assert_eq!(cfg.mel_dim, 80);
        assert_eq!(cfg.dropout, 0.1);
        assert_eq!(cfg.enc_emb_dim, 192);
        assert_eq!(cfg.enc_dim, 128);
        assert_eq!(cfg.enc_channels, vec![256, 256, 256, 256, 768]);
        assert_eq!(cfg.enc_kernel_sizes, vec![5, 3, 3, 3, 1]);
        assert_eq!(cfg.enc_dilations, vec![1, 2, 3, 4, 1]);
        assert_eq!(cfg.enc_attention_channels, 64);
        assert_eq!(cfg.enc_res2net_scale, 2);
        assert_eq!(cfg.enc_se_channels, 64);
    }

    #[test]
    fn test_bigvgan_config_values() {
        let json = r#"{
            "mel_dim": 80,
            "upsample_initial_channel": 1536,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "upsample_rates": [5, 3, 2, 2, 2, 2],
            "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4]
        }"#;
        let cfg: TokenizerV1DecoderBigVGANConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.mel_dim, 80);
        assert_eq!(cfg.upsample_initial_channel, 1536);
        assert_eq!(cfg.resblock_kernel_sizes, vec![3, 7, 11]);
        assert_eq!(cfg.resblock_dilation_sizes, vec![vec![1, 3, 5]; 3]);
        assert_eq!(cfg.upsample_rates, vec![5, 3, 2, 2, 2, 2]);
        assert_eq!(cfg.upsample_kernel_sizes, vec![11, 7, 4, 4, 4, 4]);
    }

    #[test]
    fn test_top_level_config_defaults() {
        let json = r#"{
            "decoder_config": {
                "dit_config": {},
                "bigvgan_config": {
                    "mel_dim": 80,
                    "upsample_initial_channel": 1536,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [5, 3, 2, 2, 2, 2],
                    "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4]
                }
            }
        }"#;
        let cfg: TokenizerV1Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.input_sample_rate, 24000);
        assert_eq!(cfg.output_sample_rate, 24000);
        assert_eq!(cfg.decode_upsample_rate, 1920);
        assert_eq!(cfg.encode_downsample_rate, 1920);
    }

    #[test]
    fn test_config_deserialize_override() {
        let json = r#"{"n_mels": 64, "n_head": 8}"#;
        let cfg: TokenizerV1EncoderConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.n_mels, 64);
        assert_eq!(cfg.n_head, 8);
        // rest should be defaults
        assert_eq!(cfg.n_ctx, 1500);
        assert_eq!(cfg.n_state, 1280);
    }

    #[test]
    fn test_full_config_nesting() {
        let json = r#"{
            "encoder_config": {"n_mels": 64},
            "decoder_config": {
                "dit_config": {"hidden_size": 512},
                "bigvgan_config": {
                    "mel_dim": 80,
                    "upsample_initial_channel": 1536,
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [5, 3, 2, 2, 2, 2],
                    "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4]
                }
            },
            "input_sample_rate": 16000
        }"#;
        let cfg: TokenizerV1Config = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.encoder_config.n_mels, 64);
        assert_eq!(cfg.encoder_config.n_ctx, 1500); // default
        assert_eq!(cfg.decoder_config.dit_config.hidden_size, 512);
        assert_eq!(cfg.decoder_config.dit_config.num_hidden_layers, 22); // default
        assert_eq!(cfg.input_sample_rate, 16000);
        assert_eq!(cfg.output_sample_rate, 24000); // default
    }
}
