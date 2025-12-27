//! Pooled SQLite connection

use di::inject;
use di::injectable;
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use std::env;
use std::ops::{Deref, DerefMut};
use std::sync::Mutex;

pub struct DatabaseConnection {
    connection: SqlitePool,
}

/// Global shared pool for testing - allows tests to pre-populate data that the
/// DI-created DatabaseConnection will see. This is a workaround for the more-di
/// framework not supporting pool injection.
static TEST_POOL: Mutex<Option<SqlitePool>> = Mutex::new(None);

impl DatabaseConnection {
    /// Set a shared test pool that will be used instead of creating a new one.
    /// Must be called before any DatabaseConnection is created via DI.
    pub fn set_test_pool(pool: SqlitePool) {
        let mut guard = TEST_POOL.lock().unwrap();
        *guard = Some(pool);
    }

    /// Clear the test pool. Call this during test cleanup.
    pub fn clear_test_pool() {
        let mut guard = TEST_POOL.lock().unwrap();
        *guard = None;
    }
}

#[injectable]
impl DatabaseConnection {
    #[inject]
    pub fn create() -> DatabaseConnection {
        // If a test pool is set, use it instead of creating a new connection
        {
            let guard = TEST_POOL.lock().unwrap();
            if let Some(pool) = guard.as_ref() {
                return DatabaseConnection {
                    connection: pool.clone(),
                };
            }
        }

        dotenvy::dotenv().ok();
        let mut connection_string = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

        // For tests, add shared cache mode so multiple connections see the same data
        if env::var("DATABASE_SHARED_CACHE").is_ok() && !connection_string.contains("cache=shared")
        {
            if connection_string.contains('?') {
                connection_string.push_str("&cache=shared");
            } else {
                connection_string.push_str("?cache=shared");
            }
        }

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_lazy(&connection_string)
            .expect("Cannot connect to database");

        DatabaseConnection { connection: pool }
    }

    /// Create from an existing pool (for testing)
    pub fn from_pool(pool: SqlitePool) -> Self {
        DatabaseConnection { connection: pool }
    }
}

impl Deref for DatabaseConnection {
    type Target = SqlitePool;

    fn deref(&self) -> &Self::Target {
        &self.connection
    }
}

impl DerefMut for DatabaseConnection {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.connection
    }
}
