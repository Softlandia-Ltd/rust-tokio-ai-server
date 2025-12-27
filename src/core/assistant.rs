//! LLM Assistant service.
//!

use crate::infrastructure::entities;
use log::{debug, info};
use minijinja::context;
use nalgebra::DVector;
use std::fmt::Display;
use std::str::FromStr;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tokio::sync::mpsc;
use tokio::time::Instant;
use uuid::timestamp::context;
use wgcore::gpu::GpuInstance;
use wgcore::kernel::CommandEncoderExt;
use wgcore::shapes::ViewShapeBuffers;
use wgml::gguf::Gguf;
use wgml::models::gpt2::Gpt2Tokenizer;
use wgml::models::llama2::cpu::Llama2Config;
use wgml::models::llama2::{Llama2, Llama2State, Llama2Weights, LlamaModelType, LlamaTokenizer};

pub struct InferenceTask {
    messages: Vec<ChatMessage>,
    return_channel: mpsc::Sender<String>,
}

impl InferenceTask {
    pub fn new(messages: Vec<ChatMessage>) -> (InferenceTask, mpsc::Receiver<String>) {
        let (sender, receiver) = mpsc::channel::<String>(1000);

        (
            InferenceTask {
                messages,
                return_channel: sender,
            },
            receiver,
        )
    }

    pub fn as_jinja_input(&self) -> minijinja::Value {
        let messages: Vec<minijinja::Value> =
            self.messages.iter().map(|m| m.as_jinja_value()).collect();

        minijinja::context! {
            messages => messages
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    role: Role,
    content: String,
}

impl ChatMessage {
    pub fn as_jinja_value(&self) -> minijinja::Value {
        minijinja::context! {
            role => match self.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            },
            content => self.content
        }
    }
}

#[derive(Debug, Clone)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl From<entities::Message> for ChatMessage {
    fn from(m: entities::Message) -> Self {
        Self {
            content: m.text,
            role: match m.kind {
                entities::MessageKind::System => Role::System,
                entities::MessageKind::User => Role::User,
                entities::MessageKind::Bot => Role::Assistant,
            },
        }
    }
}

pub async fn background_task(mut task_queue: mpsc::Receiver<InferenceTask>) -> () {
    let model_file_name = std::env::var("MODEL_FILE_NAME")
        .unwrap_or("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf".to_owned());
    let context_size = std::env::var("CONTEXT_SIZE")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
        .unwrap_or(32_768);

    println!("Loading model: {}", model_file_name);

    let gguf_file = File::open(model_file_name)
        .await
        .expect("failed to open model file");
    let gguf_start_time = Instant::now();
    let gguf_mmap = unsafe { memmap2::Mmap::map(&gguf_file) }.expect("failed to map file");
    let gguf = Gguf::from_bytes(&gguf_mmap[..]).expect("bad gguf");
    info!(
        "GGUF model loaded in {:.2} seconds.",
        gguf_start_time.elapsed().as_secs_f32()
    );

    let gpu = GpuInstance::new()
    .await
    .expect("failed to create GPU");
    let device = gpu.device();
    info!("GPU device created.");
    info!("GPU device features: {:?}", device.features());

    let chat_template_str = gguf
        .metadata
        .get("tokenizer.chat_template")
        .map(|v| v.as_string().to_owned())
        .unwrap_or("chat template missing".into());

    let transformer =
        Llama2::new(device, LlamaModelType::Llama).expect("failed to create LlamaModel");

    let mut config = Llama2Config::from_gguf(&gguf);
    config.seq_len = config.seq_len.min(context_size);
    let weights = Llama2Weights::from_gguf(device, &config, &gguf);
    let tokenizer = Gpt2Tokenizer::from_gguf(&gguf);
    let state = Llama2State::new(device, &config);

    let mut chat_template_env = minijinja::Environment::new();
    chat_template_env.set_trim_blocks(true);
    chat_template_env.add_global("bos_token", tokenizer.bos_str());
    chat_template_env.add_global("eos_token", tokenizer.eos_str());
    chat_template_env.add_global("add_generation_prompt", true);
    chat_template_env
        .add_template("main", &chat_template_str)
        .unwrap();
    let chat_template = chat_template_env.get_template("main").unwrap();

    let view_shapes = ViewShapeBuffers::new();

    loop {
        match task_queue.recv().await {
            None => {
                return;
            }
            Some(task) => {
                // Run the transformer.
                let prompt_str = chat_template.render(task.as_jinja_input()).unwrap();

                let prompt_tokens = tokenizer.encode(&prompt_str);
                let mut token = prompt_tokens[0];
                let mut logits = DVector::zeros(config.vocab_size);
                view_shapes.clear_tmp();

                let inference_start = Instant::now();
                let mut prefill_time = Instant::now();
                let mut total_generated = 0;

                for pos in 0.. {
                    let is_prefill = pos < prompt_tokens.len() - 1;

                    let (rope_config, rms_norm_config, attn_params) =
                        config.derived_configs(pos as u32);

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

                    if token < (config.vocab_size / 2) {
                        state.x.copy_from_view(
                            &mut encoder,
                            weights.token_embd.column(token as u32),
                        );
                    } else {
                        state.x.copy_from_view(
                            &mut encoder,
                            weights
                                .token_embd
                                .column((token - config.vocab_size / 2) as u32),
                        );
                    }

                    if pos % 50 == 0 {
                        if is_prefill {
                            println!("Prefilling token {pos}");
                        } else {
                            println!("Generating token {pos}");
                        }
                    }

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
                        pos as u32,
                    );
                    drop(compute_pass);

                    if !is_prefill {
                        state
                            .logits_readback()
                            .copy_from(&mut encoder, state.logits());

                        gpu.queue().submit(Some(encoder.finish()));

                        state
                            .logits_readback()
                            .read_to(gpu.device(), logits.as_mut_slice())
                            .await
                            .unwrap();
                    } else {
                        gpu.queue().submit(Some(encoder.finish()));
                    }

                    let mut sampler = wgml::models::sampler::Sampler::new(logits.len(), 0.9, 0.95);

                    if pos + 1 >= prompt_tokens.len() {
                        let next_token = sampler.sample(&mut logits);

                        if next_token == tokenizer.eos() {
                            break;
                        } else {
                            let token_str = tokenizer.decode(&[next_token as u32]);

                            match task.return_channel.send(token_str).await {
                                Ok(_) => {}
                                Err(_) => break,
                            }
                        }

                        token = next_token;
                        total_generated += 1;
                    } else {
                        token = prompt_tokens[pos + 1];

                        prefill_time = Instant::now();
                    }
                }

                let inference_end = Instant::now();
                let total_duration = inference_end - inference_start;
                let prefill_duration = prefill_time - inference_start;
                let generation_duration = total_duration - prefill_duration;

                println!(
                    "Inference done, total time: {total_duration:?} for {total_generated} tokens."
                );
                println!(
                    "Prefill time: {prefill_duration:?}, or {:.2} tokens/s",
                    (prompt_tokens.len() as f32) / prefill_duration.as_secs_f32()
                );
                println!(
                    "Generation time: {generation_duration:?} or {:.2} tokens/s",
                    (total_generated as f32) / generation_duration.as_secs_f32()
                );
            }
        }
    }
}

pub async fn forward(transformer: &Llama2) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::infrastructure::entities;
    use chrono::Utc;
    use uuid::Uuid;

    #[test]
    fn test_chat_message_from_user_entity() {
        let user_message = entities::Message {
            id: Uuid::new_v4(),
            conversation_id: Uuid::new_v4(),
            kind: entities::MessageKind::User,
            created_at: Utc::now(),
            text: "Hello".to_string(),
        };

        let chat_message: ChatMessage = user_message.into();
        assert!(matches!(chat_message.role, Role::User));
        assert_eq!(chat_message.content, "Hello");
    }

    #[test]
    fn test_chat_message_from_bot_entity() {
        let bot_message = entities::Message {
            id: Uuid::new_v4(),
            conversation_id: Uuid::new_v4(),
            kind: entities::MessageKind::Bot,
            created_at: Utc::now(),
            text: "Hi there!".to_string(),
        };

        let chat_message: ChatMessage = bot_message.into();
        assert!(matches!(chat_message.role, Role::Assistant));
        assert_eq!(chat_message.content, "Hi there!");
    }

    #[test]
    fn test_chat_message_from_system_entity() {
        let system_message = entities::Message {
            id: Uuid::new_v4(),
            conversation_id: Uuid::new_v4(),
            kind: entities::MessageKind::System,
            created_at: Utc::now(),
            text: "You are an assistant".to_string(),
        };

        let chat_message: ChatMessage = system_message.into();
        assert!(matches!(chat_message.role, Role::System));
        assert_eq!(chat_message.content, "You are an assistant");
    }

    #[test]
    fn test_chat_message_as_jinja_value() {
        let message = ChatMessage {
            role: Role::User,
            content: "Test message".to_string(),
        };

        let jinja_val = message.as_jinja_value();
        // Verify structure without full minijinja inspection
        assert!(jinja_val.as_object().is_some());
    }

    #[tokio::test]
    async fn test_inference_task_new_creates_channel() {
        let messages = vec![ChatMessage {
            role: Role::User,
            content: "Hello".to_string(),
        }];

        let (task, mut receiver) = InferenceTask::new(messages);

        // Should be able to send a token
        task.return_channel.send("test".to_string()).await.unwrap();

        // Should be able to receive it
        let received = receiver.recv().await;
        assert_eq!(received, Some("test".to_string()));
    }

    #[tokio::test]
    async fn test_inference_task_as_jinja_input() {
        let messages = vec![
            ChatMessage {
                role: Role::System,
                content: "System prompt".to_string(),
            },
            ChatMessage {
                role: Role::User,
                content: "User message".to_string(),
            },
        ];

        let (task, _) = InferenceTask::new(messages);
        let jinja_input = task.as_jinja_input();

        // Verify it's structured properly
        assert!(jinja_input.as_object().is_some());
    }
}
