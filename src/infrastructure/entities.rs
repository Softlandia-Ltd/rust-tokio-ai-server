//! Database entities

use chrono::{DateTime, Utc};
use sqlx::FromRow;
use uuid::Uuid;

#[derive(Debug, Clone, FromRow)]
pub struct Conversation {
    pub id: Uuid,
    pub user: Uuid,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, sqlx::Type)]
#[repr(u8)]
pub enum MessageKind {
    System = 1,
    Bot = 2,
    User = 3,
}

#[derive(Debug, Clone, FromRow)]
pub struct Message {
    pub id: Uuid,
    pub conversation_id: Uuid,
    pub kind: MessageKind,
    pub created_at: DateTime<Utc>,
    pub text: String,
}
