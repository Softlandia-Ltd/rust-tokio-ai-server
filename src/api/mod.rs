use async_trait::async_trait;
use axum::extract::FromRequestParts;
use axum::http::StatusCode;
use axum::http::request::Parts;
use std::str::FromStr;
use uuid::Uuid;

pub mod conversations;

const X_USER_ID: &str = "X-User-ID";

#[derive(Debug)]
pub struct ExtractUser(pub Uuid);

#[async_trait]
impl<S> FromRequestParts<S> for ExtractUser
where
    S: Send + Sync,
{
    type Rejection = (StatusCode, &'static str);

    async fn from_request_parts(
        parts: &mut Parts,
        _state: &S,
    ) -> Result<Self, (StatusCode, &'static str)> {
        if let Some(user_id) = parts.headers.get(X_USER_ID) {
            let user_id = user_id
                .to_str()
                .map_err(|_| (StatusCode::BAD_REQUEST, "invalid user id"))?;
            let user_id = Uuid::from_str(user_id)
                .map_err(|_| (StatusCode::BAD_REQUEST, "invalid user id"))?;
            Ok(ExtractUser(user_id))
        } else {
            Err((StatusCode::BAD_REQUEST, "`X-User-ID` header is missing"))
        }
    }
}
