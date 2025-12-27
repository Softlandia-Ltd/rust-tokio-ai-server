//! Conversations endpoints

use crate::TASK_SENDER;
use crate::api::ExtractUser;
use crate::api::conversations::schemas::{ConversationList, CreateConversation, CreateMessage};
use crate::core::assistant::{ChatMessage, InferenceTask};
use crate::core::traits::ConversationService;
use anyhow::anyhow;
use async_stream::stream;
use axum::extract::Path;
use axum::http::StatusCode;
use axum::response::Sse;
use axum::response::sse::{Event, KeepAlive};
use axum::routing::{get, post};
use axum::{Json, Router};
use di::Ref;
use di_axum::Inject;
use futures_util::Stream;
use std::convert::Infallible;
use std::time::Duration;
use uuid::Uuid;

pub fn router() -> Router {
    Router::new()
        .route("/", get(list_conversations).post(new_conversation))
        .route(
            "/:id/messages",
            get(conversation_messages).post(post_message),
        )
}

async fn list_conversations(
    Inject(conversation_service): Inject<dyn ConversationService>,
    ExtractUser(current_user): ExtractUser,
) -> (StatusCode, Json<ConversationList>) {
    let conversations = conversation_service.list_conversations(current_user).await;

    (
        StatusCode::OK,
        ConversationList {
            conversations: conversations
                .into_iter()
                .map(schemas::Conversation::from)
                .collect(),
        }
        .into(),
    )
}

async fn new_conversation(
    Inject(conversation_service): Inject<dyn ConversationService>,
    ExtractUser(current_user): ExtractUser,
    Json(create_conversation): Json<CreateConversation>,
) -> Sse<impl Stream<Item = Result<Event, &'static str>>> {
    let conversation = conversation_service.create_conversation(current_user).await;

    save_message_and_generate_response(
        conversation_service,
        current_user,
        conversation.id,
        create_conversation.message,
    )
    .await
}

async fn conversation_messages(
    Inject(conversation_service): Inject<dyn ConversationService>,
    Path(conversation_id): Path<Uuid>,
    ExtractUser(current_user): ExtractUser,
) -> (StatusCode, Json<schemas::MessagesList>) {
    let messages = conversation_service
        .list_messages(current_user, conversation_id)
        .await;

    if let Ok(messages) = messages {
        (
            StatusCode::OK,
            Json(schemas::MessagesList {
                messages: messages.into_iter().map(schemas::Message::from).collect(),
            }),
        )
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(schemas::MessagesList::default()),
        )
    }
}

async fn post_message(
    Inject(conversation_service): Inject<dyn ConversationService>,
    ExtractUser(current_user): ExtractUser,
    Path(conversation_id): Path<Uuid>,
    Json(message): Json<schemas::CreateMessage>,
) -> Sse<impl Stream<Item = Result<Event, &'static str>>> {
    save_message_and_generate_response(
        conversation_service,
        current_user,
        conversation_id,
        message.text,
    )
    .await
}

async fn save_message_and_generate_response(
    conversation_service: Ref<dyn ConversationService>,
    current_user: Uuid,
    conversation_id: Uuid,
    message: String,
) -> Sse<impl Stream<Item = Result<Event, &'static str>> + Sized> {
    match conversation_service
        .create_user_message(current_user, conversation_id, message)
        .await
    {
        Ok(message) => {
            let message_id = Uuid::new_v4();
            let conversation_id = message.conversation_id.clone();

            let conversation_messages = conversation_service
                .list_messages(current_user, conversation_id)
                .await
                .expect("failed to list user messages");

            let chat_messages = conversation_messages
                .into_iter()
                .map(ChatMessage::from)
                .collect();

            let (task, mut receiver) = InferenceTask::new(chat_messages);

            let task_sender = TASK_SENDER.get().expect("TASK_SENDER should be set");

            task_sender.send(task).await.unwrap();

            let stream = stream! {
                yield Ok(Event::default().event("new_message").json_data(schemas::Message::from(message)).unwrap());

                let mut assistant_message = String::new();

                while let Some(message_part) = receiver.recv().await {
                    assistant_message.push_str(&message_part);
                    yield Ok(Event::default().event("message_part").retry(Duration::from_millis(100)).json_data(schemas::MessagePart {
                        conversation_id,
                        message_id,
                        message_part
                    }).expect("REASON"));
                }


                conversation_service
                    .create_bot_message_with_id(current_user, conversation_id, assistant_message, message_id)
                    .await.expect("failed to save assistant message! this is bad");
            };

            Sse::new(stream).keep_alive(KeepAlive::default())
        }
        Err(_) => panic!(),
    }
}

pub mod schemas {
    use crate::infrastructure::entities;
    use chrono::{DateTime, Utc};
    use serde::{Deserialize, Serialize};
    use uuid::Uuid;

    #[derive(Deserialize, Debug)]
    pub struct CreateConversation {
        pub message: String,
    }

    #[derive(Serialize, Debug)]
    pub struct Conversation {
        pub id: Uuid,
        pub created_at: DateTime<Utc>,
        pub title: Option<String>,
    }

    impl From<entities::Conversation> for Conversation {
        fn from(conversation: entities::Conversation) -> Self {
            Conversation {
                id: conversation.id,
                created_at: conversation.created_at,
                title: None,
            }
        }
    }

    #[derive(Serialize, Debug)]
    pub struct ConversationList {
        pub conversations: Vec<Conversation>,
    }

    #[derive(Serialize, Debug, Default)]
    pub struct MessagesList {
        pub messages: Vec<Message>,
    }

    #[derive(Serialize, Debug)]
    pub enum MessageKind {
        System,
        Bot,
        User,
    }

    impl From<entities::MessageKind> for MessageKind {
        fn from(kind: entities::MessageKind) -> Self {
            match kind {
                entities::MessageKind::System => MessageKind::System,
                entities::MessageKind::Bot => MessageKind::Bot,
                entities::MessageKind::User => MessageKind::User,
            }
        }
    }

    #[derive(Serialize, Debug)]
    pub struct Message {
        pub conversation_id: Uuid,
        pub id: Uuid,
        pub kind: MessageKind,
        pub text: String,
        pub created_at: DateTime<Utc>,
    }

    impl From<entities::Message> for Message {
        fn from(message: entities::Message) -> Self {
            Message {
                conversation_id: message.conversation_id,
                id: message.id,
                kind: message.kind.into(),
                text: message.text,
                created_at: message.created_at,
            }
        }
    }

    #[derive(Deserialize, Debug)]
    pub struct CreateMessage {
        pub text: String,
    }

    #[derive(Serialize, Debug)]
    pub struct MessagePart {
        pub conversation_id: Uuid,
        pub message_id: Uuid,
        pub message_part: String,
    }
}
