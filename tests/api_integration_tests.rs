//! API Integration Tests
//!
//! Tests the HTTP API endpoints with a real database.
//! These tests focus on the read-only API endpoints that don't require
//! the LLM inference backend.
//!
//! Tests are serialized because they share a global test pool.
//!
//! Note: The `more-di` DI framework doesn't support injecting custom pools.
//! We work around this by using `DatabaseConnection::set_test_pool()` to set
//! a global pool that the DI-created DatabaseConnection will use.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use chrono::Utc;
use di::{Injectable, ServiceCollection};
use di_axum::RouterServiceProviderExtensions;
use serde_json::Value;
use serial_test::serial;
use sqlx::SqlitePool;
use std::sync::atomic::{AtomicU32, Ordering};
use tokio_local_llm_api::{
    api, core::services::MyConversationService, infrastructure::database::DatabaseConnection,
    infrastructure::repositories::DbConversationRepository,
};
use tower::ServiceExt;
use uuid::Uuid;

/// Counter for unique test database URIs
static TEST_DB_COUNTER: AtomicU32 = AtomicU32::new(0);

/// Setup test database with migrations and returns pool
/// Uses in-memory SQLite for test isolation
async fn setup_test_db() -> SqlitePool {
    let db_num = TEST_DB_COUNTER.fetch_add(1, Ordering::SeqCst);
    // Use file URI format with shared cache - each test gets a unique DB
    let db_url = format!("sqlite:file:testdb{}?mode=memory&cache=shared", db_num);

    let pool = SqlitePool::connect(&db_url).await.unwrap();
    sqlx::migrate!().run(&pool).await.unwrap();

    // Set this pool as the global test pool so DI uses it
    DatabaseConnection::set_test_pool(pool.clone());

    pool
}

/// Clean up after test
fn cleanup_test_db() {
    DatabaseConnection::clear_test_pool();
}

/// Create test app - uses the global test pool set by setup_test_db()
fn create_test_app() -> axum::Router {
    let provider = ServiceCollection::new()
        .add(DatabaseConnection::transient())
        .add(DbConversationRepository::scoped())
        .add(MyConversationService::scoped())
        .build_provider()
        .unwrap();

    axum::Router::new()
        .nest("/conversations", api::conversations::router())
        .with_provider(provider)
}

#[tokio::test]
#[serial]
async fn test_list_conversations_empty() {
    let _pool = setup_test_db().await;

    let app = create_test_app();

    let user_id = Uuid::new_v4();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/conversations")
                .header("X-User-ID", user_id.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(json["conversations"].as_array().unwrap().len(), 0);

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_list_conversations_requires_auth() {
    let _pool = setup_test_db().await;

    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/conversations")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Should fail without X-User-ID header
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_get_messages_nonexistent_conversation() {
    let _pool = setup_test_db().await;

    let app = create_test_app();

    let user_id = Uuid::new_v4();
    let fake_conversation_id = Uuid::new_v4();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/conversations/{}/messages", fake_conversation_id))
                .header("X-User-ID", user_id.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // API returns 200 OK with empty messages for non-existent conversation
    // (The query just returns no rows, not an error)
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["messages"].as_array().unwrap().len(), 0);

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_get_messages_wrong_user() {
    let pool = setup_test_db().await;

    let owner = Uuid::new_v4();
    let other_user = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();

    // Create conversation owned by 'owner' - use Uuid directly, same as production code
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id)
        .bind(owner)
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/conversations/{}/messages", conversation_id))
                .header("X-User-ID", other_user.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // The query JOINs on user_id, so different user sees empty results, not an error
    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["messages"].as_array().unwrap().len(), 0);

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_get_messages_success() {
    let pool = setup_test_db().await;

    let user_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();
    let message_id = Uuid::new_v4();

    // Create conversation - bind Uuid directly to match production code
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id)
        .bind(user_id)
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    // Create a message - bind Uuid directly to match production code
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, kind, created_at, text) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(message_id)
    .bind(conversation_id)
    .bind(3) // User message
    .bind(Utc::now().to_rfc3339())
    .bind("Hello!")
    .execute(&pool)
    .await
    .unwrap();

    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri(format!("/conversations/{}/messages", conversation_id))
                .header("X-User-ID", user_id.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let messages = json["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["text"], "Hello!");

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_list_conversations_with_data() {
    let pool = setup_test_db().await;

    let user_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();

    // Create a conversation for this user - bind Uuid directly to match production code
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id)
        .bind(user_id)
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    let app = create_test_app();

    let response = app
        .oneshot(
            Request::builder()
                .uri("/conversations")
                .header("X-User-ID", user_id.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();

    let conversations = json["conversations"].as_array().unwrap();
    assert_eq!(conversations.len(), 1);
    assert_eq!(conversations[0]["id"], conversation_id.to_string());

    cleanup_test_db();
}

#[tokio::test]
#[serial]
async fn test_user_isolation() {
    let pool = setup_test_db().await;

    let user1 = Uuid::new_v4();
    let user2 = Uuid::new_v4();

    // Create conversations for different users - bind Uuid directly to match production code
    for (user, count) in [(user1, 2), (user2, 3)] {
        for _ in 0..count {
            sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
                .bind(Uuid::new_v4())
                .bind(user)
                .bind(Utc::now().to_rfc3339())
                .execute(&pool)
                .await
                .unwrap();
        }
    }

    let app = create_test_app();

    // User1 should see 2 conversations
    let response = app
        .oneshot(
            Request::builder()
                .uri("/conversations")
                .header("X-User-ID", user1.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["conversations"].as_array().unwrap().len(), 2);

    // User2 should see 3 conversations - need new app instance since we consumed it
    let app = create_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/conversations")
                .header("X-User-ID", user2.to_string())
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["conversations"].as_array().unwrap().len(), 3);

    cleanup_test_db();
}
