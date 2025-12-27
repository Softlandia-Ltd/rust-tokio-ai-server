# Tokio Local LLM API - AI Agent Instructions

## Project Context

**This is a proof-of-concept/demo project** created as a reference implementation for GPU-accelerated LLM inference in Rust. It's not production-ready and serves primarily as educational material for presentations.

## Architecture Overview

This is a Rust-based local LLM inference server using WGPU for GPU acceleration. The application loads quantized GGUF models (like Llama-3.2-3B) and serves them via an Axum REST API with SSE streaming responses.

**Three-layer architecture:**
- **API Layer** (`src/api/`): Axum HTTP handlers and routing
- **Core Layer** (`src/core/`): Business logic services and LLM inference
- **Infrastructure Layer** (`src/infrastructure/`): Database repositories and entities

**Key architectural patterns:**
- Dependency injection via `more-di` and `more-di-axum` crates
- Background task model for LLM inference using mpsc channels
- SQLite with sqlx for conversation persistence
- Server-Sent Events (SSE) for streaming LLM responses

## Critical Components

### LLM Inference Pipeline (`src/core/assistant.rs`)
- **Background task**: Runs in separate Tokio task, consuming `InferenceTask` messages via mpsc channel
- **Model loading**: Uses `wgml` (custom WGPU ML library) to load GGUF models and `wgcore` for GPU operations
- **Chat templating**: Uses MiniJinja to format messages with model-specific chat templates from GGUF metadata
- **Streaming**: Token generation streams back via mpsc channel to API handlers
- **Environment vars**: `MODEL_FILE_NAME` (default: `models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`), `CONTEXT_SIZE` (default: 32768)

### Dependency Injection Pattern
Services are registered in `main.rs:web_server_task()`:
```rust
ServiceCollection::new()
    .add(DatabaseConnection::singleton())      // Single instance
    .add(DbConversationRepository::scoped())  // Per-request
    .add(MyConversationService::scoped())     // Per-request
```

Use `Inject<dyn TraitName>` parameter in Axum handlers to receive dependencies. Traits are defined in `src/core/traits.rs` and `src/infrastructure/traits.rs`.

### User Authentication
Authentication is **header-based only**: All API requests require `X-User-ID` header with a valid UUID. The `ExtractUser` extractor (`src/api/mod.rs`) validates this and provides the user ID to handlers.

### Database Schema
SQLite with two tables (`migrations/20250716125628_initial.up.sql`):
- `conversations`: id (TEXT/UUID), user (TEXT/UUID), created_at (TEXT/DateTime)
- `messages`: id, conversation_id, kind (INTEGER: 1=System, 2=Bot, 3=User), created_at, text

All entities use UUID primary keys. **Important:** When binding UUIDs in sqlx queries, bind the `Uuid` type directly (not `.to_string()`). SQLx handles the conversion internally, but mixing string-bound and Uuid-bound queries will cause mismatches because the internal storage format differs.

## Local Dependencies

This project depends on **local path dependencies** for GPU ML operations:
- `wgml`: Custom WGPU-based ML library (GGUF loading, Llama2 inference)
- `wgcore`: GPU compute abstractions
- `wgebra`: Linear algebra (patched in Cargo.toml)

These are referenced via Windows-style paths: `path = "..\\wgml\\crates\\wgml"`. If modifying dependencies, maintain these path references.

## Development Workflows

### Running the Server
```bash
cargo run --release  # Release mode recommended for LLM performance
```
- Server listens on `0.0.0.0:3000`
- Static frontend in `static/` directory (reference only, not actively maintained)
- API endpoints under `/conversations`

### Running Tests
```bash
cargo test                    # Run all tests
cargo test --lib             # Library tests only
cargo test --test '*'        # Integration tests only
cargo test -- --nocapture    # Show println! output
```

### Running Inference Tests
Inference tests require a GGUF model file and are ignored by default:
```bash
# Run inference tests (requires model file in models/ directory)
cargo test --test inference_tests -- --ignored

# Run all tests including ignored ones
cargo test -- --include-ignored

# Use a custom model path
MODEL_FILE_NAME=models/my-model.gguf cargo test --test inference_tests -- --ignored
```

### Database Setup
Requires `.env` file with `DATABASE_URL=sqlite:./database.db`. SQLx uses lazy connection pooling (max 5 connections). Run migrations manually with sqlx-cli if needed.

### Model Configuration
Place GGUF models in `models/` directory. Supported formats: Llama-based models with GGUF tokenizer metadata. The system auto-extracts chat templates from GGUF metadata.

## Project-Specific Conventions

### Error Handling
- Services return `Result<T, ()>` with empty error type - log errors using `log::error!("{e}")` in repository layer
- API handlers return HTTP status codes directly: `(StatusCode::OK, Json(data))`
- Use `anyhow::Result` for top-level functions

### Message Flow Pattern
Conversation endpoints follow this pattern:
1. Handler receives user message
2. Saves user message to DB via `ConversationService::create_user_message()`
3. Loads full conversation history from DB
4. Converts entities to `ChatMessage` and sends `InferenceTask` to background queue
5. Streams LLM tokens via SSE as they arrive
6. Accumulates full response and saves as bot message

See `src/api/conversations.rs:save_message_and_generate_response()` for the complete flow.

### Service Trait Pattern
Domain services are defined as async traits (`src/core/traits.rs`) with concrete implementations in `src/core/services.rs`. Use helper methods like `create_user_message()` rather than `create_raw_message()` directly.

### Repository Layer
Repositories use sqlx with raw SQL queries (no ORM). All queries include user ID checks for row-level security. Use `sqlx::query_as` for SELECT statements that map to entities.

## Testing Conventions

### Unit Tests (`src/core/assistant.rs`)
- Located in module with `#[cfg(test)] mod tests`
- Use `#[tokio::test]` for async tests
- Test entity conversions and channel behavior without mocking the DI framework

### Integration Tests (`tests/`)

**API Integration Tests (`api_integration_tests.rs`)**
- Tests HTTP endpoints using `tower::ServiceExt::oneshot()` with real Axum router
- Uses `DatabaseConnection::set_test_pool()` to inject shared test pool into DI framework
- Tests are serialized with `#[serial]` to avoid test pool conflicts
- Use in-memory SQLite with unique DB names per test: `sqlite:file:testdbN?mode=memory&cache=shared`

**Database Tests (`db_tests.rs`)**
- Tests schema, migrations, and entity storage using in-memory SQLite
- Uses `SqlitePool::connect(":memory:")` + `sqlx::migrate!().run(&pool).await`

**Auth Tests (`auth_tests.rs`)**
- Tests `ExtractUser` extractor using Axum's `FromRequestParts`

### Test Database Pool Injection
The `more-di` framework doesn't support injecting custom pools. The workaround is:
```rust
// In DatabaseConnection (src/infrastructure/database.rs)
static TEST_POOL: Mutex<Option<SqlitePool>> = Mutex::new(None);

impl DatabaseConnection {
    pub fn set_test_pool(pool: SqlitePool) { ... }
    pub fn clear_test_pool() { ... }
}
```
Call `set_test_pool()` after creating the test DB, before creating the app router.

### Running Tests
```bash
cargo test                              # Run all tests
cargo test --lib                        # Library unit tests only
cargo test --test api_integration_tests # API integration tests
cargo test --test db_tests              # Database tests
```

### Testing Pitfalls
- **UUID binding**: Always bind `Uuid` type directly, not `.to_string()` - sqlx uses different internal formats
- The `more-di` framework's `Ref<T>` type makes service-layer unit testing with mocks challenging
- Prefer integration tests with real SQLite for repository/service behavior

## Common Pitfalls

- **Channel blocking**: The inference task queue has capacity 10 - sending will block if full
- **WGPU backend**: Currently hardcoded to Vulkan (`Backends::VULKAN`) - may need adjustment for non-Vulkan systems
- **Text-only responses**: System prompt explicitly forbids Markdown/HTML - LLM outputs plain text only
- **Windows paths**: Cargo.toml uses Windows backslash paths for local dependencies - use raw strings or double backslashes
- **UUID binding mismatch**: SQLx converts UUIDs internally - mixing `.to_string()` and `Uuid` binds causes query mismatches
