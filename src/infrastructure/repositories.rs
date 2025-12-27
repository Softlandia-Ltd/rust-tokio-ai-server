//! DB Repository abstractions

use crate::infrastructure::database::DatabaseConnection;
use crate::infrastructure::entities::{Conversation, Message};
use crate::infrastructure::traits::ConversationRepository;
use async_trait::async_trait;
use chrono::Utc;
use di::{Ref, injectable};
use log::error;
use uuid::Uuid;

#[injectable(ConversationRepository)]
pub struct DbConversationRepository {
    connection: Ref<DatabaseConnection>,
}

#[async_trait]
impl ConversationRepository for DbConversationRepository {
    async fn list_conversations(&self, user_id: Uuid) -> Result<Vec<Conversation>, ()> {
        sqlx::query_as(
            "SELECT * FROM conversations WHERE user = ? ORDER BY datetime(created_at) ASC",
        )
        .bind(user_id)
        .fetch_all(&**self.connection)
        .await
        .map_err(|e| error!("{e}"))
    }

    async fn create_conversation(&self, conversation: Conversation) -> Result<Conversation, ()> {
        sqlx::query_as(
            "INSERT INTO conversations (id, user, created_at) VALUES (?, ?, ?) RETURNING *",
        )
        .bind(conversation.id)
        .bind(conversation.user)
        .bind(conversation.created_at)
        .fetch_one(&**self.connection)
        .await
        .map_err(|e| error!("{e}"))
    }

    async fn delete_conversation(&self, conversation_id: Uuid) -> Result<(), ()> {
        todo!()
    }

    async fn list_conversation_messages(
        &self,
        user_id: Uuid,
        conversation: Uuid,
    ) -> Result<Vec<Message>, ()> {
        sqlx::query_as(
            "SELECT messages.id, messages.conversation_id, messages.created_at, messages.kind, messages.text FROM messages INNER JOIN conversations ON conversations.id = messages.conversation_id WHERE conversation_id = ? AND user = ? ORDER BY datetime(messages.created_at) ASC",
        )
            .bind(conversation)
            .bind(user_id)
            .fetch_all(&**self.connection)
            .await
            .map_err(|e| error!("{e}"))
    }

    async fn create_message_in_conversation(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: Message,
    ) -> Result<Message, ()> {
        // TODO: check user id
        sqlx::query_as(
            "INSERT INTO messages (id, conversation_id, kind, created_at, text) VALUES (?, ?, ?, ?, ?) RETURNING *",
        )
            .bind(Uuid::new_v4())
            .bind(conversation_id)
            .bind(message.kind)
            .bind(message.created_at)
            .bind(message.text)
            .fetch_one(&**self.connection)
            .await
            .map_err(|e| error!("{e}"))
    }
}
