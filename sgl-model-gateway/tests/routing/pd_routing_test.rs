//! Prefill/Decode (PD) routing integration tests
//!
//! Tests for prefill-decode disaggregation routing mode.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::Request,
    http::{header::CONTENT_TYPE, StatusCode},
};
use serde_json::json;
use smg::{
    config::RouterConfig,
    core::{BasicWorkerBuilder, Worker, WorkerType as SmgWorkerType},
    routers::RouterFactory,
};
use tower::ServiceExt;

use crate::common::{
    mock_worker::{
        take_generate_payloads_for_port, HealthStatus, MockWorker, MockWorkerConfig,
        WorkerType,
    },
    test_app, AppTestContext, TestWorkerConfig, create_test_context,
};

#[cfg(test)]
mod pd_routing_tests {
    use super::*;

    /// Test basic PD mode routing with prefill and decode workers
    #[tokio::test]
    async fn test_pd_mode_basic_routing() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![
                    ("http://127.0.0.1:19800".to_string(), None),
                    ("http://127.0.0.1:19801".to_string(), None),
                ],
                vec![
                    "http://127.0.0.1:19802".to_string(),
                    "http://127.0.0.1:19803".to_string(),
                ],
            )
            .power_of_two_policy(1)
            .host("127.0.0.1")
            .port(3800)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        // Note: For PD mode tests, we need to start prefill and decode workers separately
        // The test context will need to handle this specially
        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                // Prefill workers
                TestWorkerConfig::prefill(19800),
                TestWorkerConfig::prefill(19801),
                // Decode workers
                TestWorkerConfig::decode(19802),
                TestWorkerConfig::decode(19803),
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Send requests and verify they succeed
        for i in 0..10 {
            let payload = json!({
                "text": format!("PD mode request {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::OK,
                "PD mode request should succeed"
            );
        }

        ctx.shutdown().await;
    }

    /// Test PD mode with round robin policy
    #[tokio::test]
    async fn test_pd_mode_round_robin() {
        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19810".to_string(), None)],
                vec![
                    "http://127.0.0.1:19811".to_string(),
                    "http://127.0.0.1:19812".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3801)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19810),
                TestWorkerConfig::decode(19811),
                TestWorkerConfig::decode(19812),
            ],
        )
        .await;

        let app = ctx.create_app().await;
        let mut success_count = 0;

        for i in 0..20 {
            let payload = json!({
                "text": format!("PD round robin {}", i),
                "stream": false
            });

            let req = Request::builder()
                .method("POST")
                .uri("/generate")
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_string(&payload).unwrap()))
                .unwrap();

            let resp = app.clone().oneshot(req).await.unwrap();
            if resp.status() == StatusCode::OK {
                success_count += 1;
            }
        }

        assert_eq!(
            success_count, 20,
            "All requests should succeed in PD mode with round robin"
        );

        ctx.shutdown().await;
    }

    /// Test PD mode handles worker failures gracefully
    #[tokio::test]
    async fn test_pd_mode_with_failing_decode_worker() {
        use smg::config::RetryConfig;

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![("http://127.0.0.1:19820".to_string(), None)],
                vec![
                    "http://127.0.0.1:19821".to_string(),
                    "http://127.0.0.1:19822".to_string(),
                ],
            )
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3802)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .retry_config(RetryConfig {
                max_retries: 3,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
                ..Default::default()
            })
            .build_unchecked();

        let ctx = AppTestContext::new_with_config(
            config,
            vec![
                TestWorkerConfig::prefill(19820),
                MockWorkerConfig {
                    port: 19821,
                    worker_type: WorkerType::Decode,
                    health_status: HealthStatus::Healthy,
                    response_delay_ms: 0,
                    fail_rate: 1.0, // Failing decode worker
                },
                TestWorkerConfig::decode(19822), // Healthy decode worker
            ],
        )
        .await;

        let app = ctx.create_app().await;

        // Request should succeed via retry to healthy decode worker
        let payload = json!({
            "text": "Test with failing decode worker",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "Request should succeed via retry to healthy decode worker"
        );

        ctx.shutdown().await;
    }

    /// Test host-centric KV mode sends prefill first, then decode, with a shared host_kv_id.
    #[tokio::test]
    async fn test_pd_host_kv_pool_injects_shared_host_kv_id() {
        std::env::set_var("NO_PROXY", "127.0.0.1,localhost");
        std::env::set_var("no_proxy", "127.0.0.1,localhost");

        let mut prefill_mock = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Prefill,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let mut decode_mock = MockWorker::new(MockWorkerConfig {
            port: 0,
            worker_type: WorkerType::Decode,
            health_status: HealthStatus::Healthy,
            response_delay_ms: 0,
            fail_rate: 0.0,
        });
        let prefill_url = prefill_mock.start().await.unwrap();
        let decode_url = decode_mock.start().await.unwrap();
        let prefill_port: u16 = prefill_url.rsplit(':').next().unwrap().parse().unwrap();
        let decode_port: u16 = decode_url.rsplit(':').next().unwrap().parse().unwrap();

        let config = RouterConfig::builder()
            .prefill_decode_mode(
                vec![(prefill_url.clone(), Some(9100))],
                vec![decode_url.clone()],
            )
            .pd_host_kv_pool(true)
            .round_robin_policy()
            .host("127.0.0.1")
            .port(3810)
            .max_payload_size(256 * 1024 * 1024)
            .request_timeout_secs(600)
            .worker_startup_timeout_secs(5)
            .worker_startup_check_interval_secs(1)
            .max_concurrent_requests(64)
            .queue_timeout_secs(60)
            .build_unchecked();

        let app_context = create_test_context(config).await;
        let prefill_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(prefill_url)
                .worker_type(SmgWorkerType::Prefill {
                    bootstrap_port: Some(9100),
                })
                .build(),
        );
        let decode_worker: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new(decode_url)
                .worker_type(SmgWorkerType::Decode)
                .build(),
        );
        app_context.worker_registry.register(prefill_worker);
        app_context.worker_registry.register(decode_worker);

        let router = RouterFactory::create_router(&app_context).await.unwrap();
        let app = test_app::create_test_app_with_context(Arc::from(router), app_context);

        let payload = json!({
            "text": "host kv route smoke",
            "stream": false
        });

        let req = Request::builder()
            .method("POST")
            .uri("/generate")
            .header(CONTENT_TYPE, "application/json")
            .body(Body::from(serde_json::to_string(&payload).unwrap()))
            .unwrap();

        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            status,
            StatusCode::OK,
            "host-kv PD request should succeed, body={}",
            String::from_utf8_lossy(&body)
        );

        let prefill_payloads = take_generate_payloads_for_port(prefill_port);
        let decode_payloads = take_generate_payloads_for_port(decode_port);
        assert_eq!(prefill_payloads.len(), 1);
        assert_eq!(decode_payloads.len(), 1);

        let prefill_payload = &prefill_payloads[0];
        let decode_payload = &decode_payloads[0];
        let prefill_host_kv_id = prefill_payload
            .get("host_kv_id")
            .and_then(|v| v.as_str())
            .expect("prefill payload should include host_kv_id");
        let decode_host_kv_id = decode_payload
            .get("host_kv_id")
            .and_then(|v| v.as_str())
            .expect("decode payload should include host_kv_id");

        assert_eq!(prefill_host_kv_id, decode_host_kv_id);
        assert!(!prefill_host_kv_id.is_empty());
        assert!(prefill_payload.get("bootstrap_host").is_some());
        assert!(prefill_payload.get("bootstrap_port").is_some());
        assert!(prefill_payload.get("bootstrap_room").is_some());
        assert_eq!(
            prefill_payload.get("bootstrap_room"),
            decode_payload.get("bootstrap_room")
        );

        prefill_mock.stop().await;
        decode_mock.stop().await;
    }
}
