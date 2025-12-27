//! Infrastructure traits, used for DI on higher levels

use crate::infrastructure::entities;
use async_trait::async_trait;
use uuid::Uuid;

#[async_trait]
pub trait ConversationRepository: Send + Sync {
    async fn list_conversations(&self, user_id: Uuid) -> Result<Vec<entities::Conversation>, ()>;
    async fn create_conversation(
        &self,
        conversation: entities::Conversation,
    ) -> Result<entities::Conversation, ()>;

    async fn delete_conversation(&self, conversation_id: Uuid) -> Result<(), ()>;

    async fn list_conversation_messages(
        &self,
        user_id: Uuid,
        conversation: Uuid,
    ) -> Result<Vec<entities::Message>, ()>;

    async fn create_message_in_conversation(
        &self,
        user_id: Uuid,
        conversation_id: Uuid,
        message: entities::Message,
    ) -> Result<entities::Message, ()>;
}
