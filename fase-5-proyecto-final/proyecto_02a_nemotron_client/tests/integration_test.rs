use proyecto_02a_nemotron_client::{
    ChatMessage, NemotronClient, NemotronConfig, Role, SseEvent, SseParser, StreamTelemetry,
};

#[test]
fn test_types_serialization_and_roles() {
    let msg_user = ChatMessage::user("¿Cómo funciona el KV Cache?");
    assert_eq!(msg_user.role, Role::User);
    assert_eq!(msg_user.content, "¿Cómo funciona el KV Cache?");

    let msg_sys = ChatMessage::system("Instrucción del sistema");
    assert_eq!(msg_sys.role, Role::System);

    let json = serde_json::to_string(&msg_user).unwrap();
    assert!(json.contains("\"role\":\"user\""));
    assert!(json.contains("KV Cache"));
}

#[test]
fn test_sse_parser_tokens_and_done() {
    // 1. Línea con token válido
    let sample_line = r#"data: {"id":"chat-1","model":"nemotron","choices":[{"index":0,"delta":{"content":"Hola"}}]}"#;
    let event = SseParser::parse_line(sample_line).unwrap();
    assert_eq!(event, SseEvent::Token("Hola".to_string()));

    // 2. Línea de finalización [DONE]
    let done_line = "data: [DONE]";
    let event_done = SseParser::parse_line(done_line).unwrap();
    assert_eq!(event_done, SseEvent::Done);

    // 3. Comentario SSE (keep-alive)
    let comment_line = ": ping";
    let event_comment = SseParser::parse_line(comment_line).unwrap();
    assert_eq!(event_comment, SseEvent::Comment("ping".to_string()));

    // 4. Línea vacía
    let empty_line = "   ";
    let event_empty = SseParser::parse_line(empty_line).unwrap();
    assert_eq!(event_empty, SseEvent::Empty);
}

#[test]
fn test_stream_telemetry_metrics() {
    let mut telemetry = StreamTelemetry::new();
    std::thread::sleep(std::time::Duration::from_millis(15));
    telemetry.record_chunk("Hola ");
    telemetry.record_chunk("mundo ");
    telemetry.record_chunk("Rust!");
    telemetry.finish();

    assert!(telemetry.ttft_ms().is_some());
    assert!(telemetry.ttft_ms().unwrap() >= 10.0);
    assert_eq!(telemetry.token_count, 3);
    assert_eq!(telemetry.character_count, "Hola mundo Rust!".chars().count());
    assert!(telemetry.tokens_per_second() > 0.0);
}

#[tokio::test]
async fn test_mock_simulator_sync_completion() {
    let config = NemotronConfig::for_mock_simulator();
    let client = NemotronClient::new(config).unwrap();

    let messages = vec![
        ChatMessage::system("Eres Nemotron"),
        ChatMessage::user("Explícame optimización en Rust"),
    ];

    let response = client.complete(&messages).await.unwrap();
    assert_eq!(response.choices.len(), 1);
    let text = &response.choices[0].message.content;
    assert!(text.contains("Rust"));
    assert!(response.usage.is_some());
}

#[tokio::test]
async fn test_mock_simulator_streaming_tokens() {
    let config = NemotronConfig::for_mock_simulator();
    let client = NemotronClient::new(config).unwrap();

    let messages = vec![
        ChatMessage::user("¿Qué sabes de arquitecturas GPU?"),
    ];

    let mut rx = client.stream_chat(&messages).await.unwrap();
    let mut collected = String::new();
    let mut chunks_count = 0;

    while let Some(res) = rx.recv().await {
        let token = res.unwrap();
        collected.push_str(&token);
        chunks_count += 1;
    }

    assert!(chunks_count > 5);
    assert!(collected.contains("NVIDIA") || collected.contains("GPU") || collected.contains("Hopper"));
}
