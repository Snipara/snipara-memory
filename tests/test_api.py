from __future__ import annotations

from fastapi.testclient import TestClient

from snipara_memory import InMemoryMemoryStore, MemoryService, create_app


def test_health_and_store_recall_flow() -> None:
    app = create_app(MemoryService(store=InMemoryMemoryStore()))
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    store = client.post(
        "/v1/namespaces/demo/memories",
        json={
            "title": "Webhook policy",
            "content": "Stripe webhooks require signature verification and idempotency.",
        },
    )
    assert store.status_code == 200

    recall = client.post(
        "/v1/namespaces/demo/memories/recall",
        json={"query": "How do we handle Stripe webhooks?"},
    )
    assert recall.status_code == 200
    body = recall.json()
    assert len(body) == 1
    assert body[0]["memory"]["title"] == "Webhook policy"
