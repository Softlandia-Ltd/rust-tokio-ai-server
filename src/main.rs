//! Local LLM web server using WGPU
//!
//! (c) Softlandia 2025

use tokio_local_llm_api::TASK_SENDER;
use tokio_local_llm_api::api;
use tokio_local_llm_api::core;
use tokio_local_llm_api::core::assistant::{ChatMessage, InferenceTask};
use tokio_local_llm_api::core::services::MyConversationService;
use tokio_local_llm_api::core::traits::ConversationService;
use tokio_local_llm_api::infrastructure::database::DatabaseConnection;
use tokio_local_llm_api::infrastructure::repositories::DbConversationRepository;

use anyhow::anyhow;
use axum::http::{HeaderValue, Method};
use axum::response::Html;
use axum::{
    Json, Router,
    http::StatusCode,
    routing::{get, post},
};
use di::{
    InjectBuilder, Injectable, ServiceCollection, ServiceLifetime, ServiceProvider, injectable,
};
use di_axum::RouterServiceProviderExtensions;
use log::info;
use serde::{Deserialize, Serialize};
use teloxide::handler;
use teloxide::prelude::*;
use teloxide::types::{MediaKind, MessageKind};
use teloxide::types::{MediaText, ParseMode};
use tokio::runtime::{Builder, Runtime};
use tokio::sync::{OnceCell, mpsc};
use tokio::task;
use tokio::task::JoinHandle;
use tower::ServiceBuilder;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use uuid::Uuid;

const ENABLE_TELEGRAM_HANDLER: bool = false;

fn main() -> anyhow::Result<()> {
    // initialize tracing
    tracing_subscriber::fmt::init();

    let runtime: Runtime = Builder::new_multi_thread().enable_all().build()?;

    // background task for local LLM
    let (task_sender, task_receiver) = mpsc::channel(10);
    let assistant_join_handle = runtime.spawn(core::assistant::background_task(task_receiver));
    TASK_SENDER
        .set(task_sender)
        .expect("task sender should not be set");

    let web_task_handle = runtime.spawn(web_server_task());

    runtime.block_on(async {
        web_task_handle
            .await
            .expect("failed to join web_task_handle");
        assistant_join_handle
            .await
            .expect("failed to join assistant_join_handle");
    });

    Ok(())
}

async fn web_server_task() {
    let provider = ServiceCollection::new()
        .add(DatabaseConnection::singleton())
        .add(DbConversationRepository::scoped())
        .add(MyConversationService::scoped())
        .build_provider()
        .unwrap();

    // build our application with a route
    let app = Router::new()
        .route("/", get(index))
        .nest_service(
            "/static",
            ServiceBuilder::new().service(ServeDir::new("static")),
        )
        .nest("/conversations", api::conversations::router())
        .layer(
            CorsLayer::new()
                .allow_headers(Any)
                .allow_methods([Method::GET, Method::POST])
                .allow_origin([
                    "http://localhost:3000".parse::<HeaderValue>().unwrap(),
                    "http://localhost:5173".parse::<HeaderValue>().unwrap(),
                ]),
        )
        .with_provider(provider);

    // run our app with hyper, listening globally on port 3000
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    info!("listening on {}", listener.local_addr().unwrap());
    axum::serve(listener, app).await.unwrap();
    info!("Shutting down...");
}

async fn index() -> Html<&'static str> {
    Html(include_str!("../static/index.html"))
}
