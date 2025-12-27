//! Database and schema tests
//!
//! Tests SQLite migrations, entity storage, and schema constraints

use chrono::Utc;
use sqlx::SqlitePool;
use uuid::Uuid;

/// Setup test database with migrations
async fn setup_test_db() -> SqlitePool {
    let pool = SqlitePool::connect(":memory:").await.unwrap();
    sqlx::migrate!().run(&pool).await.unwrap();
    pool
}

#[tokio::test]
async fn test_database_migrations_work() {
    // This test verifies migrations apply successfully
    let pool = setup_test_db().await;

    // Verify tables exist
    let result = sqlx::query("SELECT name FROM sqlite_master WHERE type='table'")
        .fetch_all(&pool)
        .await
        .unwrap();

    assert!(result.len() >= 2); // Should have conversations and messages tables
}

#[tokio::test]
async fn test_uuid_storage_in_sqlite() {
    let pool = setup_test_db().await;

    let user_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();

    // Insert as TEXT
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id.to_string())
        .bind(user_id.to_string())
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    // Retrieve and parse back
    let row: (String, String, String) =
        sqlx::query_as("SELECT id, user, created_at FROM conversations WHERE id = ?")
            .bind(conversation_id.to_string())
            .fetch_one(&pool)
            .await
            .unwrap();

    assert_eq!(Uuid::parse_str(&row.0).unwrap(), conversation_id);
    assert_eq!(Uuid::parse_str(&row.1).unwrap(), user_id);
}

#[tokio::test]
async fn test_message_kind_enum_storage() {
    use tokio_local_llm_api::infrastructure::entities::MessageKind;

    let pool = setup_test_db().await;

    let conversation_id = Uuid::new_v4();
    let user_id = Uuid::new_v4();

    // Create conversation first
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id.to_string())
        .bind(user_id.to_string())
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    // Test all message kinds
    for (kind, value) in [
        (MessageKind::System, 1),
        (MessageKind::Bot, 2),
        (MessageKind::User, 3),
    ] {
        let msg_id = Uuid::new_v4();
        sqlx::query("INSERT INTO messages (id, conversation_id, kind, created_at, text) VALUES (?, ?, ?, ?, ?)")
            .bind(msg_id.to_string())
            .bind(conversation_id.to_string())
            .bind(value)
            .bind(Utc::now().to_rfc3339())
            .bind(format!("Test {:?}", kind))
            .execute(&pool)
            .await
            .unwrap();
    }

    // Verify all were stored
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM messages WHERE conversation_id = ?")
        .bind(conversation_id.to_string())
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(count.0, 3);
}

#[tokio::test]
async fn test_conversation_cascade_delete() {
    let pool = setup_test_db().await;

    let user_id = Uuid::new_v4();
    let conversation_id = Uuid::new_v4();

    // Create conversation
    sqlx::query("INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?)")
        .bind(conversation_id.to_string())
        .bind(user_id.to_string())
        .bind(Utc::now().to_rfc3339())
        .execute(&pool)
        .await
        .unwrap();

    // Create message
    sqlx::query(
        "INSERT INTO messages (id, conversation_id, kind, created_at, text) VALUES (?, ?, ?, ?, ?)",
    )
    .bind(Uuid::new_v4().to_string())
    .bind(conversation_id.to_string())
    .bind(3) // User message
    .bind(Utc::now().to_rfc3339())
    .bind("Test")
    .execute(&pool)
    .await
    .unwrap();

    // Delete conversation (should cascade to messages)
    sqlx::query("DELETE FROM conversations WHERE id = ?")
        .bind(conversation_id.to_string())
        .execute(&pool)
        .await
        .unwrap();

    // Verify messages were deleted
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM messages WHERE conversation_id = ?")
        .bind(conversation_id.to_string())
        .fetch_one(&pool)
        .await
        .unwrap();

    assert_eq!(count.0, 0);
}
