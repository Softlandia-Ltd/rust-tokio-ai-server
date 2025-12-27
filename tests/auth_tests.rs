//! Unit tests for API authentication extractor

use axum::extract::FromRequestParts;
use axum::http::{Request, StatusCode};
use tokio_local_llm_api::api::ExtractUser;
use uuid::Uuid;

#[tokio::test]
async fn test_extract_user_valid_uuid() {
    let user_id = Uuid::new_v4();
    let mut req = Request::builder()
        .header("X-User-ID", user_id.to_string())
        .body(())
        .unwrap();

    let (mut parts, _) = req.into_parts();
    let result = ExtractUser::from_request_parts(&mut parts, &()).await;

    assert!(result.is_ok());
    assert_eq!(result.unwrap().0, user_id);
}

#[tokio::test]
async fn test_extract_user_missing_header() {
    let mut req = Request::builder().body(()).unwrap();

    let (mut parts, _) = req.into_parts();
    let result = ExtractUser::from_request_parts(&mut parts, &()).await;

    assert!(result.is_err());
    let (status, message) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(message.contains("missing"));
}

#[tokio::test]
async fn test_extract_user_invalid_uuid() {
    let mut req = Request::builder()
        .header("X-User-ID", "not-a-uuid")
        .body(())
        .unwrap();

    let (mut parts, _) = req.into_parts();
    let result = ExtractUser::from_request_parts(&mut parts, &()).await;

    assert!(result.is_err());
    let (status, message) = result.unwrap_err();
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(message.contains("invalid"));
}

#[tokio::test]
async fn test_extract_user_invalid_utf8() {
    use axum::http::HeaderValue;

    let mut req = Request::builder().body(()).unwrap();
    req.headers_mut()
        .insert("X-User-ID", HeaderValue::from_bytes(&[0xFF, 0xFE]).unwrap());

    let (mut parts, _) = req.into_parts();
    let result = ExtractUser::from_request_parts(&mut parts, &()).await;

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().0, StatusCode::BAD_REQUEST);
}
