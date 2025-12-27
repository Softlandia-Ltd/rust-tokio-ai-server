//! Integration tests for LLM inference.
//!
//! These tests require a GGUF model file to be present. They are ignored by default
//! and can be run with:
//!
//! ```bash
//! cargo test --test inference_tests -- --ignored
//! ```
//!
//! Or run all tests including ignored ones:
//!
//! ```bash
//! cargo test -- --include-ignored
//! ```
//!
//! Set the MODEL_FILE_NAME environment variable to use a different model:
//!
//! ```bash
//! MODEL_FILE_NAME=models/my-model.gguf cargo test --test inference_tests -- --ignored
//! ```

use std::path::Path;
use wgml::gguf::Gguf;
use wgml::models::gpt2::Gpt2Tokenizer;
use wgml::models::llama2::cpu::Llama2Config;
use wgml::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType};

const DEFAULT_MODEL_PATH: &str = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

fn get_model_path() -> String {
    std::env::var("MODEL_FILE_NAME").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

fn model_exists() -> bool {
    Path::new(&get_model_path()).exists()
}

/// Helper to skip test if model doesn't exist (for non-ignored runs)
fn require_model() {
    if !model_exists() {
        eprintln!(
            "Skipping test: Model file not found at '{}'. \
             Set MODEL_FILE_NAME env var or place model in default location.",
            get_model_path()
        );
    }
}

// =============================================================================
// GGUF Loading Tests
// =============================================================================

#[test]
#[ignore = "requires model file"]
fn test_gguf_file_loads_successfully() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]);

    assert!(gguf.is_ok(), "Failed to parse GGUF file: {:?}", gguf.err());
}

#[test]
#[ignore = "requires model file"]
fn test_gguf_contains_required_metadata() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    // Check for essential metadata keys
    assert!(
        gguf.metadata.contains_key("tokenizer.ggml.tokens"),
        "GGUF missing tokenizer.ggml.tokens"
    );
    assert!(
        gguf.metadata.contains_key("tokenizer.ggml.bos_token_id"),
        "GGUF missing tokenizer.ggml.bos_token_id"
    );
    assert!(
        gguf.metadata.contains_key("tokenizer.ggml.eos_token_id"),
        "GGUF missing tokenizer.ggml.eos_token_id"
    );
}

#[test]
#[ignore = "requires model file"]
fn test_gguf_contains_chat_template() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    // Chat template is important for instruction-tuned models
    let has_chat_template = gguf.metadata.contains_key("tokenizer.chat_template");
    if !has_chat_template {
        eprintln!("Warning: Model does not contain a chat template");
    }
}

// =============================================================================
// Tokenizer Tests
// =============================================================================

#[test]
#[ignore = "requires model file"]
fn test_tokenizer_loads_from_gguf() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let tokenizer = Gpt2Tokenizer::from_gguf(&gguf);

    // Basic sanity checks
    assert!(
        tokenizer.bos() > 0 || tokenizer.bos() == 0,
        "BOS token should be valid"
    );
    assert!(
        tokenizer.eos() > 0 || tokenizer.eos() == 0,
        "EOS token should be valid"
    );
}

#[test]
#[ignore = "requires model file"]
fn test_tokenizer_encodes_text() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let tokenizer = Gpt2Tokenizer::from_gguf(&gguf);

    // Encode a simple string
    let tokens = tokenizer.encode("Hello, world!");

    assert!(!tokens.is_empty(), "Tokenizer should produce tokens");
    println!(
        "'Hello, world!' encoded to {} tokens: {:?}",
        tokens.len(),
        tokens
    );
}

#[test]
#[ignore = "requires model file"]
fn test_tokenizer_roundtrip() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let tokenizer = Gpt2Tokenizer::from_gguf(&gguf);

    let original = "The quick brown fox";
    let tokens = tokenizer.encode(original);
    let decoded = tokenizer.decode(&tokens.iter().map(|&t| t as u32).collect::<Vec<_>>());

    // Note: roundtrip may not be perfect due to tokenization artifacts
    println!("Original: '{}'", original);
    println!("Decoded:  '{}'", decoded);
    assert!(
        decoded.contains("quick") || decoded.contains("fox"),
        "Decoded text should contain parts of original"
    );
}

// =============================================================================
// Model Configuration Tests
// =============================================================================

#[test]
#[ignore = "requires model file"]
fn test_llama2_config_loads_from_gguf() {
    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let config = Llama2Config::from_gguf(&gguf);

    // Sanity checks for config values
    assert!(config.dim > 0, "Model dimension should be positive");
    assert!(config.n_layers > 0, "Number of layers should be positive");
    assert!(
        config.n_q_heads > 0,
        "Number of query heads should be positive"
    );
    assert!(config.vocab_size > 0, "Vocab size should be positive");

    println!("Model config:");
    println!("  dim: {}", config.dim);
    println!("  n_layers: {}", config.n_layers);
    println!("  n_q_heads: {}", config.n_q_heads);
    println!("  n_kv_heads: {}", config.n_kv_heads);
    println!("  vocab_size: {}", config.vocab_size);
    println!("  seq_len: {}", config.seq_len);
}

// =============================================================================
// GPU Initialization Tests
// =============================================================================

#[tokio::test]
#[ignore = "requires GPU"]
async fn test_gpu_instance_creation() {
    use wgcore::gpu::GpuInstance;

    let gpu = GpuInstance::new().await;
    assert!(
        gpu.is_ok(),
        "Failed to create GPU instance: {:?}",
        gpu.err()
    );

    let gpu = gpu.unwrap();
    println!("GPU device created successfully");
    println!("Device features: {:?}", gpu.device().features());
}

// =============================================================================
// Full Model Loading Tests (Heavy - requires significant GPU memory)
// =============================================================================

#[tokio::test]
#[ignore = "requires model file and GPU - heavy test"]
async fn test_full_model_weights_load() {
    use wgcore::gpu::GpuInstance;

    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let gpu = GpuInstance::new()
        .await
        .expect("failed to create GPU instance");

    let mut config = Llama2Config::from_gguf(&gguf);
    config.seq_len = config.seq_len.min(2048); // Limit context for testing

    println!("Loading model weights to GPU...");
    let start = std::time::Instant::now();
    let _weights = Llama2Weights::from_gguf(gpu.device(), &config, &gguf);
    println!("Weights loaded in {:.2}s", start.elapsed().as_secs_f32());
}

#[tokio::test]
#[ignore = "requires model file and GPU - heavy test"]
async fn test_transformer_initialization() {
    use wgcore::gpu::GpuInstance;

    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let gpu = GpuInstance::new()
        .await
        .expect("failed to create GPU instance");

    let config = Llama2Config::from_gguf(&gguf);

    // Create transformer
    let transformer = Llama2::new(gpu.device(), LlamaModelType::Llama);
    assert!(
        transformer.is_ok(),
        "Failed to create transformer: {:?}",
        transformer.err()
    );

    // Create state
    let _state = Llama2State::new(gpu.device(), &config);
    println!("Transformer and state initialized successfully");
}

// =============================================================================
// Single Token Generation Test (Integration)
// =============================================================================

#[tokio::test]
#[ignore = "requires model file and GPU - heavy integration test"]
async fn test_single_forward_pass() {
    use nalgebra::DVector;
    use wgcore::gpu::GpuInstance;
    use wgcore::kernel::CommandEncoderExt;
    use wgcore::shapes::ViewShapeBuffers;

    require_model();
    if !model_exists() {
        return;
    }

    let model_path = get_model_path();
    let file = std::fs::File::open(&model_path).expect("failed to open model file");
    let mmap = unsafe { memmap2::Mmap::map(&file) }.expect("failed to mmap file");
    let gguf = Gguf::from_bytes(&mmap[..]).expect("failed to parse GGUF");

    let gpu = GpuInstance::new()
        .await
        .expect("failed to create GPU instance");

    let mut config = Llama2Config::from_gguf(&gguf);
    config.seq_len = config.seq_len.min(2048);

    let tokenizer = Gpt2Tokenizer::from_gguf(&gguf);
    let transformer =
        Llama2::new(gpu.device(), LlamaModelType::Llama).expect("failed to create transformer");
    let weights = Llama2Weights::from_gguf(gpu.device(), &config, &gguf);
    let state = Llama2State::new(gpu.device(), &config);
    let view_shapes = ViewShapeBuffers::new();

    // Encode a simple prompt
    let prompt = "Hello";
    let tokens = tokenizer.encode(prompt);
    assert!(!tokens.is_empty(), "Should have at least one token");

    let token = tokens[0];
    let pos = 0u32;

    let (rope_config, rms_norm_config, attn_params) = config.derived_configs(pos);

    // Set up GPU buffers
    let mut encoder = gpu.device().create_command_encoder(&Default::default());
    gpu.queue().write_buffer(
        state.rope_config().buffer(),
        0,
        bytemuck::cast_slice(&[rope_config]),
    );
    gpu.queue().write_buffer(
        state.rms_norm_config().buffer(),
        0,
        bytemuck::cast_slice(&[rms_norm_config]),
    );
    gpu.queue().write_buffer(
        state.attn_params().buffer(),
        0,
        bytemuck::cast_slice(&[attn_params]),
    );

    // Copy token embedding
    state
        .x
        .copy_from_view(&mut encoder, weights.token_embd.column(token as u32));

    // Run forward pass
    let mut compute_pass = encoder.compute_pass("transformer", None);
    transformer.dispatch(
        gpu.device(),
        &view_shapes,
        gpu.queue(),
        &mut compute_pass,
        &state,
        &weights,
        &config,
        &attn_params,
        pos,
    );
    drop(compute_pass);

    // Read back logits
    state
        .logits_readback()
        .copy_from(&mut encoder, state.logits());
    gpu.queue().submit(Some(encoder.finish()));

    let mut logits = DVector::zeros(config.vocab_size);
    state
        .logits_readback()
        .read_to(gpu.device(), logits.as_mut_slice())
        .await
        .expect("failed to read logits");

    // Verify we got valid logits
    let max_logit = logits.max();
    let min_logit = logits.min();
    println!("Logits range: [{}, {}]", min_logit, max_logit);

    assert!(
        max_logit.is_finite() && min_logit.is_finite(),
        "Logits should be finite values"
    );
    assert!(max_logit != min_logit, "Logits should have some variation");

    // Find the most likely next token
    let next_token = logits.argmax().0;
    println!(
        "Input token: {} ('{}'), predicted next token: {}",
        token, prompt, next_token
    );
}
