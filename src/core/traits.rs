//! DI "Interfaces"

use crate::infrastructure::entities;
use crate::infrastructure::entities::MessageKind;
use async_trait::async_trait;
use uuid::Uuid;

#[async_trait]
pub trait ConversationService: Send + Sync {
    /// Lists all conversations for the given user.
    async fn list_conversations(&self, user_id: Uuid) -> Vec<entities::Conversation>;

    /// Creates a new conversation for the given user.
    async fn create_conversation(&self, user_id: Uuid) -> entities::Conversation;

    /// Deletes a given conversation from the given user.
    ///
    /// Returns `Err` if the conversation did not exist or the user didn't have permissions to
    /// delete it.
    async fn delete_conversation(&self, user_id: Uuid) -> Result<(), ()>;

    /// List all messages in a conversation.
    ///
    /// Returns `Err` if the user doesn't have permissions to view this conversation.
    async fn list_messages(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
    ) -> Result<Vec<entities::Message>, ()>;

    /// Creates a new message in a conversation.
    ///
    /// The helper functions `create_X_message` should be used instead for clarity.
    async fn create_raw_message(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        kind: MessageKind,
        content: String,
        message_id: Uuid,
    ) -> Result<entities::Message, ()>;

    /// Create a new user message in a conversation.
    ///
    /// Returns `Err` if conversation does not exist or the user doesn't have permissions to post
    /// to it.
    async fn create_user_message(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: String,
    ) -> Result<entities::Message, ()> {
        self.create_raw_message(
            user_id,
            conversation_id,
            MessageKind::User,
            message,
            Uuid::new_v4(),
        )
        .await
    }

    /// Create a new bot message in a conversation.
    ///
    /// Returns `Err` if the conversation doesn't exist.
    async fn create_bot_message(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: String,
    ) -> Result<entities::Message, ()> {
        self.create_raw_message(
            user_id,
            conversation_id,
            MessageKind::Bot,
            message,
            Uuid::new_v4(),
        )
        .await
    }

    async fn create_bot_message_with_id(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: String,
        message_id: Uuid,
    ) -> Result<entities::Message, ()> {
        self.create_raw_message(
            user_id,
            conversation_id,
            MessageKind::Bot,
            message,
            message_id,
        )
        .await
    }

    /// Create a new system message in a conversation.
    ///
    /// Returns `Err` if the conversation doesn't exist.
    async fn create_system_message(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: String,
    ) -> Result<entities::Message, ()> {
        self.create_raw_message(
            user_id,
            conversation_id,
            MessageKind::System,
            message,
            Uuid::new_v4(),
        )
        .await
    }
}
