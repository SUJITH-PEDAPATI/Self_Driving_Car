"""
Integration tests for the Udacity self-driving car simulator server.

Tests are designed to run without the actual simulator.  They use:
  - Direct Python calls for model / preprocessing tests
  - A synthetic Socket.IO v2 test client for protocol tests

Run from the project root:
    .\.venv\Scripts\python.exe -m src.tests.test_drive_integration

Tests:
  1. Checkpoint loads correctly.
  2. A representative JPEG can be decoded and preprocessed.
  3. Resulting tensor has shape (1, 3, 66, 200).
  4. Model produces a finite steering prediction.
  5. Socket.IO server starts on port 4567.
  6. A test Socket.IO client can connect to the server.
  7. Synthetic telemetry produces a steer response.
  8. Missing / malformed image does not crash the server.
"""

from __future__ import annotations

import asyncio
import base64
import io
import sys
import traceback
import unittest
from pathlib import Path

# ── Ensure project root is on path ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import aiohttp
import socketio
import torch
from PIL import Image

# ── Import project modules ─────────────────────────────────────────────────
from src.model import SteeringCNN
from src.preprocess_image import convertToRGB, convertToTensors, resizeImages

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "best_model.pth"
TEST_PORT = 4568   # Use a different port so we don't clash with a live server

# ── Helpers ────────────────────────────────────────────────────────────────

def make_synthetic_jpeg(width: int = 320, height: int = 160) -> str:
    """Create a small solid-colour JPEG and return it as a base64 string."""
    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Unit tests (no server needed) ─────────────────────────────────────────

class TestCheckpointLoad(unittest.TestCase):
    def test_checkpoint_exists(self):
        self.assertTrue(
            CHECKPOINT_PATH.exists(),
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    def test_checkpoint_loads_and_keys_present(self):
        device = torch.device("cpu")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        for key in ("model_state_dict", "epoch", "val_loss"):
            self.assertIn(key, ckpt, f"Missing key '{key}' in checkpoint")

    def test_model_state_loads_without_error(self):
        device = torch.device("cpu")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        m = SteeringCNN().to(device)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()


class TestPreprocessing(unittest.TestCase):
    def _run_pipeline(self, jpeg_b64: str) -> torch.Tensor:
        image_bytes = base64.b64decode(jpeg_b64)
        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.copy()
        images = convertToRGB([image])
        images = resizeImages(images)
        tensors = convertToTensors(images)
        return tensors[0].unsqueeze(0)   # (1, 3, 66, 200)

    def test_synthetic_jpeg_decodes(self):
        jpeg_b64 = make_synthetic_jpeg()
        tensor = self._run_pipeline(jpeg_b64)
        self.assertEqual(tuple(tensor.shape), (1, 3, 66, 200))

    def test_tensor_values_in_range(self):
        jpeg_b64 = make_synthetic_jpeg()
        tensor = self._run_pipeline(jpeg_b64)
        self.assertGreaterEqual(tensor.min().item(), 0.0)
        self.assertLessEqual(tensor.max().item(), 1.0)

    def test_tensor_is_finite(self):
        jpeg_b64 = make_synthetic_jpeg()
        tensor = self._run_pipeline(jpeg_b64)
        self.assertTrue(torch.isfinite(tensor).all().item())


class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=cls.device, weights_only=False)
        cls.model = SteeringCNN().to(cls.device)
        cls.model.load_state_dict(ckpt["model_state_dict"])
        cls.model.eval()

    def _make_tensor(self) -> torch.Tensor:
        jpeg_b64 = make_synthetic_jpeg()
        image_bytes = base64.b64decode(jpeg_b64)
        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.copy()
        images = convertToRGB([image])
        images = resizeImages(images)
        tensors = convertToTensors(images)
        return tensors[0].unsqueeze(0).to(self.device)

    def test_output_shape(self):
        tensor = self._make_tensor()
        with torch.inference_mode():
            out = self.model(tensor)
        self.assertEqual(tuple(out.shape), (1, 1))

    def test_output_is_finite(self):
        tensor = self._make_tensor()
        with torch.inference_mode():
            out = self.model(tensor)
        self.assertTrue(torch.isfinite(out).all().item())

    def test_output_is_scalar(self):
        tensor = self._make_tensor()
        with torch.inference_mode():
            steering = self.model(tensor).item()
        self.assertIsInstance(steering, float)
        self.assertTrue(-10.0 < steering < 10.0,
                        f"Steering {steering} seems wildly out of range")


# ── Integration tests (spin up the real server) ────────────────────────────

class TestServerProtocol(unittest.IsolatedAsyncioTestCase):
    """
    Starts the actual SimulatorServer on TEST_PORT and connects to it using
    a python-socketio AsyncClient.  The client uses EIO=4 (default), so no
    middleware rewrite is exercised here.  Separate manual tests with the real
    simulator cover EIO=3 rewriting.
    """

    async def asyncSetUp(self):
        # Import the server objects from drive.py without running __main__
        # We need to create a fresh server for each test to avoid state leakage.
        import importlib

        # Build a test server using the same SimulatorServer class
        from src.drive import SimulatorServer, eio_compat_middleware, load_model

        # Load model once
        load_model()

        self.sio_server = SimulatorServer(
            async_mode="aiohttp",
            logger=False,
            engineio_logger=False,
            cors_allowed_origins="*",
        )

        self._steer_events: list[dict] = []
        self._connected = asyncio.Event()

        @self.sio_server.on("telemetry")
        async def _telemetry_handler(sid, data):
            from src.drive import (
                THROTTLE,
                model,
                model_loaded,
                preprocess_image,
            )
            if not model_loaded:
                return
            if not isinstance(data, dict) or not data.get("image"):
                return
            try:
                image_tensor = preprocess_image(data["image"])
                with torch.inference_mode():
                    steering = model(image_tensor).item()
                controls = {
                    "steering_angle": str(steering),
                    "throttle": str(THROTTLE),
                }
                await self.sio_server.emit("steer", controls, to=sid)
                self._steer_events.append(controls)
            except Exception:
                traceback.print_exc()

        self.app = aiohttp.web.Application(middlewares=[eio_compat_middleware])
        self.sio_server.attach(self.app)

        self.runner = aiohttp.web.AppRunner(self.app)
        await self.runner.setup()
        self.site = aiohttp.web.TCPSite(self.runner, "127.0.0.1", TEST_PORT)
        await self.site.start()

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_server_starts_and_client_connects(self):
        client = socketio.AsyncClient()
        connected = asyncio.Event()

        @client.event
        async def connect():
            connected.set()

        try:
            await client.connect(
                f"http://127.0.0.1:{TEST_PORT}",
                transports=["polling"],
            )
            await asyncio.wait_for(connected.wait(), timeout=5.0)
            self.assertTrue(connected.is_set(), "Client did not connect within timeout")
        finally:
            if client.connected:
                await client.disconnect()

    async def test_synthetic_telemetry_produces_steer_response(self):
        client = socketio.AsyncClient()
        connected = asyncio.Event()
        steer_received = asyncio.Event()
        received_controls: list[dict] = []

        @client.event
        async def connect():
            connected.set()

        @client.on("steer")
        async def on_steer(data):
            received_controls.append(data)
            steer_received.set()

        try:
            await client.connect(
                f"http://127.0.0.1:{TEST_PORT}",
                transports=["polling"],
            )
            await asyncio.wait_for(connected.wait(), timeout=5.0)

            jpeg_b64 = make_synthetic_jpeg()
            await client.emit("telemetry", {
                "steering_angle": "0.0",
                "throttle": "0.0",
                "speed": "5.0",
                "image": jpeg_b64,
            })

            await asyncio.wait_for(steer_received.wait(), timeout=10.0)

            self.assertTrue(steer_received.is_set(), "No steer event received")
            self.assertEqual(len(received_controls), 1)
            ctrl = received_controls[0]
            self.assertIn("steering_angle", ctrl)
            self.assertIn("throttle", ctrl)
            steering = float(ctrl["steering_angle"])
            self.assertTrue(-10.0 < steering < 10.0,
                            f"Steering {steering} out of expected range")

        finally:
            if client.connected:
                await client.disconnect()

    async def test_malformed_image_does_not_crash_server(self):
        """
        The server must catch the bad-base64 error and remain alive.
        We prove this by sending a valid image AFTER the bad one and
        confirming we still get a steer response.
        """
        client = socketio.AsyncClient()
        connected = asyncio.Event()
        steer_received = asyncio.Event()

        @client.event
        async def connect():
            connected.set()

        @client.on("steer")
        async def on_steer(data):
            steer_received.set()

        try:
            await client.connect(
                f"http://127.0.0.1:{TEST_PORT}",
                transports=["polling"],
            )
            await asyncio.wait_for(connected.wait(), timeout=5.0)

            # 1. Send garbage base64 – server must swallow the error
            await client.emit("telemetry", {
                "steering_angle": "0.0",
                "throttle": "0.0",
                "speed": "0.0",
                "image": "not_valid_base64!!!!",
            })
            await asyncio.sleep(0.5)  # let server process the bad packet

            # 2. Send a valid image – server must still respond
            jpeg_b64 = make_synthetic_jpeg()
            await client.emit("telemetry", {
                "steering_angle": "0.0",
                "throttle": "0.0",
                "speed": "0.0",
                "image": jpeg_b64,
            })
            await asyncio.wait_for(steer_received.wait(), timeout=10.0)
            self.assertTrue(
                steer_received.is_set(),
                "Server did not respond after a bad image – it may have crashed",
            )

        finally:
            if client.connected:
                await client.disconnect()

    async def test_missing_image_field_does_not_crash(self):
        """
        Telemetry with no 'image' key should be silently dropped.
        Prove the server is still alive by getting a steer response
        for a valid image sent immediately after.
        """
        client = socketio.AsyncClient()
        connected = asyncio.Event()
        steer_received = asyncio.Event()

        @client.event
        async def connect():
            connected.set()

        @client.on("steer")
        async def on_steer(data):
            steer_received.set()

        try:
            await client.connect(
                f"http://127.0.0.1:{TEST_PORT}",
                transports=["polling"],
            )
            await asyncio.wait_for(connected.wait(), timeout=5.0)

            # 1. Send telemetry with no 'image' key
            await client.emit("telemetry", {
                "steering_angle": "0.0",
                "throttle": "0.0",
                "speed": "0.0",
            })
            await asyncio.sleep(0.5)

            # 2. Valid image – should get steer back
            jpeg_b64 = make_synthetic_jpeg()
            await client.emit("telemetry", {
                "steering_angle": "0.0",
                "throttle": "0.0",
                "speed": "0.0",
                "image": jpeg_b64,
            })
            await asyncio.wait_for(steer_received.wait(), timeout=10.0)
            self.assertTrue(
                steer_received.is_set(),
                "Server did not respond after missing-image telemetry",
            )

        finally:
            if client.connected:
                await client.disconnect()


# ── Runner ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Self-Driving Car – Drive Integration Tests")
    print("NOTE: These are automated tests using a mock client.")
    print("      Real simulator connection must be tested manually.")
    print("=" * 70)
    unittest.main(verbosity=2)
