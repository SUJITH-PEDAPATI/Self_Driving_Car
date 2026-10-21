"""
Udacity Self-Driving Car Simulator – Python inference server
============================================================

Architecture
------------
The Udacity simulator is a Socket.IO *client* (Socket.IO v2 / EIO=3).
This script is the Socket.IO *server*, listening on port 4567.

Protocol compatibility
----------------------
python-engineio >= 4.4 removed allow_eio3 and now hard-rejects EIO=3.
python-socketio 5.x requires an explicit namespace CONNECT packet (type 0)
before routing events, but Socket.IO v2 clients (the simulator) do NOT send
that packet for the root namespace – they send events directly.

Fix 1 – aiohttp middleware:
  Rewrites `EIO=3` → `EIO=4` in the query string before engineio sees it.

Fix 2 – SimulatorServer subclass:
  Overrides _handle_eio_connect to immediately perform the Socket.IO '/'
  namespace handshake, so events arriving without a prior '0' connect packet
  are accepted and routed correctly.
"""

from __future__ import annotations

import base64
import io
import traceback
from pathlib import Path

import socketio
from socketio import packet as _sio_packet
import torch
from aiohttp import web
from PIL import Image

from src.model import SteeringCNN
from src.preprocess_image import convertToRGB, convertToTensors, resizeImages

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path("models") / "best_model.pth"
SIMULATOR_HOST = "0.0.0.0"
SIMULATOR_PORT = 4567
THROTTLE = 0.5   # Keep at 0.0 until connection and steering are verified.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model = SteeringCNN().to(device)
model_loaded = False


def load_model() -> None:
    global model_loaded
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH.resolve()}"
        )
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model_loaded = True
    print(f"[drive] Model loaded from {CHECKPOINT_PATH}")
    print(f"[drive] Device:           {device}")
    print(f"[drive] Checkpoint epoch: {checkpoint['epoch']}")
    print(f"[drive] Validation loss:  {checkpoint['val_loss']:.6f}")


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image_data: str) -> torch.Tensor:
    """Decode a base64 JPEG, apply training preprocessing, return (1,3,66,200)."""
    image_bytes = base64.b64decode(image_data)
    with Image.open(io.BytesIO(image_bytes)) as img:
        image = img.copy()

    images = convertToRGB([image])    # crop to (0,60,320,140) + RGB
    images = resizeImages(images)     # resize to 200×66
    tensors = convertToTensors(images)
    tensor = tensors[0]               # (3, 66, 200)

    if tensor.shape != (3, 66, 200):
        raise ValueError(
            f"Unexpected tensor shape after preprocessing: {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("Image tensor contains non-finite values.")

    return tensor.unsqueeze(0).to(device)   # (1, 3, 66, 200)


# ---------------------------------------------------------------------------
# Fix 2: SimulatorServer – auto-connect Socket.IO v2 clients to '/'
# ---------------------------------------------------------------------------

class SimulatorServer(socketio.AsyncServer):
    """
    Handles Socket.IO v2 clients (the Udacity simulator).

    Socket.IO v2 does NOT send an explicit Socket.IO CONNECT packet (type 0)
    for the root namespace.  python-socketio 5.x requires that packet before
    routing events.  We detect this situation in _handle_eio_message:
    when an EVENT packet arrives for an eio_sid that has not yet completed
    the '/' namespace handshake, we perform the handshake ourselves first.

    This is deliberately lazy (triggered by the first event, not by EIO
    connect) so that v3/v4 clients that DO send '0{}' work without conflict.
    """

    async def _handle_eio_message(self, eio_sid: str, data: str) -> None:
        """Auto-connect to '/' namespace on first event if not yet connected."""
        # Decode just enough of the packet to know its type
        try:
            packet_type = int(data[0:1]) if data else -1
        except (ValueError, TypeError):
            packet_type = -1

        if packet_type == _sio_packet.EVENT:
            # Check whether this eio_sid is already registered in '/'
            if not self.manager.sid_from_eio_sid(eio_sid, "/"):
                # Socket.IO v2 client – perform namespace handshake now
                await self._handle_connect(eio_sid, "/", None)

        await super()._handle_eio_message(eio_sid, data)


# ---------------------------------------------------------------------------
# Socket.IO server instance
# ---------------------------------------------------------------------------

sio = SimulatorServer(
    async_mode="aiohttp",
    logger=True,
    engineio_logger=True,
    cors_allowed_origins="*",
)


# ---------------------------------------------------------------------------
# Socket.IO event handlers
# ---------------------------------------------------------------------------

@sio.event
async def connect(sid: str, environ: dict, auth=None) -> None:
    print(f"[drive] Simulator connected: sid={sid}")


@sio.event
async def disconnect(sid: str, reason=None) -> None:
    print(f"[drive] Simulator disconnected: sid={sid}")


@sio.on("telemetry")
async def telemetry(sid: str, data: dict) -> None:
    if not model_loaded:
        print("[drive] WARNING: model not loaded, skipping telemetry.")
        return

    if not isinstance(data, dict) or not data.get("image"):
        print("[drive] WARNING: telemetry payload missing 'image' field.")
        return

    try:
        image_tensor = preprocess_image(data["image"])

        if image_tensor.shape != (1, 3, 66, 200):
            raise ValueError(
                f"Input tensor shape mismatch: {tuple(image_tensor.shape)}"
            )

        with torch.inference_mode():
            steering_raw = model(image_tensor)

        steering = steering_raw.item()

        if not torch.isfinite(torch.tensor(steering)).item():
            raise ValueError(f"Model returned non-finite steering: {steering}")

        controls = {
            "steering_angle": str(steering),
            "throttle": str(THROTTLE),
        }
        await sio.emit("steer", controls, to=sid)

        speed = data.get("speed", "N/A")
        print(
            f"[drive] Steering: {steering:+.4f} | "
            f"Throttle: {THROTTLE:.1f} | "
            f"Speed: {speed}"
        )

    except Exception:
        print("[drive] ERROR during telemetry processing:")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Fix 1: aiohttp middleware – rewrite EIO=3 → EIO=4 in query string
# ---------------------------------------------------------------------------

@web.middleware
async def eio_compat_middleware(
    request: web.Request, handler
) -> web.StreamResponse:
    """
    The Udacity simulator sends ?EIO=3.
    python-engineio 4.4+ hard-rejects any version other than EIO=4.
    This middleware rewrites the query string before engineio inspects it.
    The rest of the Engine.IO v3 protocol is wire-compatible enough that
    the connection proceeds normally.
    """
    qs = request.query_string
    if "EIO=3" in qs:
        new_qs = qs.replace("EIO=3", "EIO=4", 1)
        new_url = request.url.with_query_string(new_qs)
        request = request.clone(rel_url=new_url.relative())
    return await handler(request)


# ---------------------------------------------------------------------------
# aiohttp application
# ---------------------------------------------------------------------------

app = web.Application(middlewares=[eio_compat_middleware])
sio.attach(app)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_model()

    print(f"\n[drive] Server ready – listening on port {SIMULATOR_PORT}")
    print(f"[drive] Throttle locked at {THROTTLE}.  Open the simulator")
    print(f"[drive] and select Autonomous Mode.\n")

    web.run_app(
        app,
        host=SIMULATOR_HOST,
        port=SIMULATOR_PORT,
        access_log=None,   # suppress per-request HTTP noise; keep only our prints
    )

