//! Local LLM web server using WGPU - Library exports for testing
//!
//! (c) Softlandia 2025

pub mod api;
pub mod core;
pub mod infrastructure;

use crate::core::assistant::InferenceTask;
use tokio::sync::OnceCell;
use tokio::sync::mpsc;

pub static TASK_SENDER: OnceCell<mpsc::Sender<InferenceTask>> = OnceCell::const_new();
