//! Pooled SQLite connection

use di::inject;
use di::injectable;
use sqlx::SqlitePool;
use sqlx::sqlite::SqlitePoolOptions;
use std::env;
use std::ops::{Deref, DerefMut};

pub struct DatabaseConnection {
    connection: SqlitePool,
}

#[injectable]
impl DatabaseConnection {
    #[inject]
    pub fn create() -> DatabaseConnection {
        dotenvy::dotenv().ok();
        let connection_string = env::var("DATABASE_URL").expect("DATABASE_URL must be set");

        let pool = SqlitePoolOptions::new()
            .max_connections(5)
            .connect_lazy(&connection_string)
            .expect("Cannot connect to database");

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
