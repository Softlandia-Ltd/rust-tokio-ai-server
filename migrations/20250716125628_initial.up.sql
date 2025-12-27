-- Add up migration script here
CREATE TABLE conversations
(
    id         TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user       TEXT NOT NULL
);

CREATE UNIQUE INDEX conversations_id ON conversations (id);

CREATE TABLE messages
(
    id              TEXT PRIMARY KEY,
    conversation_id TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    kind            INTEGER NOT NULL,
    text            TEXT    NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX messages_id ON messages (id);