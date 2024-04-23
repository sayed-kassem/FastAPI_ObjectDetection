import asyncio
from pathlib import Path
import pytest
from fastapi import status
from httpx import AsyncClient
from httpx_ws import aconnect_ws
import pytest_asyncio

from main import * 

coffee_shop_image_file = Path(__file__).parent.parent / "assets" / "coffee-shop.jpg"
detected_labels = {"person", "couch", "chair", "laptop", "dining table"}

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://127.0.0.1:5000") as client:
        yield client

@pytest.mark.asyncio
@pytest.mark.slow
class TestChapter13API:
    async def test_invalid_payload(self, client):
        response = await client.post("/object-detection", files={})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_valid_payload(self, client):
        with open(coffee_shop_image_file, "rb") as image:
            files = {"image": image}
            response = await client.post("/object-detection", files=files)
        assert response.status_code == status.HTTP_200_OK
        json = response.json()
        objects = json["objects"]
        assert len(objects) > 0
        for obj in objects:
            assert "box" in obj
            assert obj["label"] in detected_labels

@pytest.mark.asyncio
@pytest.mark.slow
class TestChapter13WebSocketobjectDetection:
    async def test_single_detection(self, client):
        async with aconnect_ws("/object-detection", client) as websocket:
            async with open(coffee_shop_image_file, "rb") as image:
                await websocket.send_bytes(image.read())
                result = await websocket.receive_json()
                objects = result["objects"]
                assert len(objects) > 0
                for obj in objects:
                    assert "box" in obj
                    assert obj["label"] in detected_labels

    async def test_backpressure(self, client):
        QUEUE_LIMIT = 10
        async with aconnect_ws("/object-detection", client) as websocket:
            async with open(coffee_shop_image_file, "rb") as image:
                bytes = image.read()
                for _ in range(QUEUE_LIMIT + 1):
                    await websocket.send_bytes(bytes)
                result = await websocket.receive_json()
                assert result is not None
                with pytest.raises(asyncio.TimeoutError):
                    await websocket.receive_json(timeout=0.1)
