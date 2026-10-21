
from __future__ import annotations

import base64
import io
import time
import traceback
from pathlib import Path

import socketio
from socketio import packet as _sio_packet
import torch
from aiohttp import web
from PIL import Image

from src.model import SteeringCNN
from src.preprocess_image import convertToRGB, convertToTensors, resizeImages

CHECKPOINT_PATH = Path("models") / "best_model.pth"
SIMULATOR_HOST = "0.0.0.0"
SIMULATOR_PORT = 4567

THROTTLE = 0.15
STUCK_SPEED_MPH = 0.5
STUCK_TIMEOUT_SECONDS = 4.0
REVERSE_THROTTLE = -0.20
REVERSE_DURATION_SECONDS = 1.5
RECOVERY_FORWARD_THROTTLE = 0.08
RECOVERY_COOLDOWN_SECONDS = 5.0

last_motion_time = time.monotonic()
recovery_until = 0.0
cooldown_until = 0.0
recovery_steering = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def preprocess_image(image_data: str) -> torch.Tensor:
    image_bytes = base64.b64decode(image_data)

    with Image.open(io.BytesIO(image_bytes)) as img:
        image = img.copy()

    images = convertToRGB([image])
    images = resizeImages(images)
    tensors = convertToTensors(images)
    tensor = tensors[0]

    if tensor.shape != (3, 66, 200):
        raise ValueError(
            f"Unexpected tensor shape after preprocessing: {tuple(tensor.shape)}"
        )

    if not torch.isfinite(tensor).all():
        raise ValueError("Image tensor contains non-finite values.")

    return tensor.unsqueeze(0).to(device)


def get_recovery_controls(speed: float, steering: float) -> dict:
    global last_motion_time
    global recovery_until, cooldown_until
    global recovery_steering

    now = time.monotonic()

    if now < recovery_until:
        return {
            "steering_angle": str(recovery_steering),
            "throttle": str(REVERSE_THROTTLE),
        }

    if recovery_until != 0.0:
        recovery_until = 0.0
        cooldown_until = now + RECOVERY_COOLDOWN_SECONDS
        last_motion_time = now
        print("[drive] Recovery maneuver completed")

    if now < cooldown_until:
        return {
            "steering_angle": str(steering),
            "throttle": str(RECOVERY_FORWARD_THROTTLE),
        }

    if speed > STUCK_SPEED_MPH:
        last_motion_time = now
        return {
            "steering_angle": str(steering),
            "throttle": str(THROTTLE),
        }

    if now - last_motion_time >= STUCK_TIMEOUT_SECONDS:
        print("[drive] Car appears stuck! Starting recovery...")

        if steering > 0.05:
            recovery_steering = -0.3
        elif steering < -0.05:
            recovery_steering = 0.3
        else:
            recovery_steering = 0.3

        recovery_until = now + REVERSE_DURATION_SECONDS

        return {
            "steering_angle": str(recovery_steering),
            "throttle": str(REVERSE_THROTTLE),
        }

    return {
        "steering_angle": str(steering),
        "throttle": str(THROTTLE),
    }


class SimulatorServer(socketio.AsyncServer):
    async def _handle_eio_message(self, eio_sid: str, data: str) -> None:
        try:
            packet_type = int(data[0:1]) if data else -1
        except (ValueError, TypeError):
            packet_type = -1

        if packet_type == _sio_packet.EVENT:
            if not self.manager.sid_from_eio_sid(eio_sid, "/"):
                await self._handle_connect(eio_sid, "/", None)

        await super()._handle_eio_message(eio_sid, data)


sio = SimulatorServer(
    async_mode="aiohttp",
    logger=True,
    engineio_logger=True,
    cors_allowed_origins="*",
)


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

        speed = float(data.get("speed", 0.0))

        controls = get_recovery_controls(
            speed=speed,
            steering=steering,
        )

        await sio.emit("steer", controls, to=sid)

        print(
            f"[drive] Speed: {speed:.2f} mph | "
            f"Steering: {controls['steering_angle']} | "
            f"Throttle: {controls['throttle']}"
        )

    except Exception:
        print("[drive] ERROR during telemetry processing:")
        traceback.print_exc()


@web.middleware
async def eio_compat_middleware(
    request: web.Request, handler
) -> web.StreamResponse:
    qs = request.query_string

    if "EIO=3" in qs:
        new_qs = qs.replace("EIO=3", "EIO=4", 1)
        new_url = request.url.with_query_string(new_qs)
        request = request.clone(rel_url=new_url.relative())

    return await handler(request)


app = web.Application(middlewares=[eio_compat_middleware])
sio.attach(app)


if __name__ == "__main__":
    load_model()

    print(f"\n[drive] Server ready – listening on port {SIMULATOR_PORT}")
    print(f"[drive] Throttle set to {THROTTLE}. Open the simulator")
    print("[drive] and select Autonomous Mode.\n")

    web.run_app(
        app,
        host=SIMULATOR_HOST,
        port=SIMULATOR_PORT,
        access_log=None,
    )
