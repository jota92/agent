import asyncio
import logging
import math
import os
import queue
import threading
from typing import Any, Awaitable, Dict
from urllib.parse import urlparse

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from pyppeteer import launch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_URL = "https://www.google.com/"
DEFAULT_VIEWPORT = {"width": 1280, "height": 720}
BUTTON_MAP = {0: "left", 1: "middle", 2: "right"}

app = Flask(__name__)
app.config["SECRET_KEY"] = "development-secret-key-change-me"
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


def normalize_url(raw_url: str) -> str:
    """Ensure user-supplied URLs include a scheme."""
    parsed = urlparse(raw_url)
    if parsed.scheme:
        return raw_url
    return f"http://{raw_url}"


class BrowserController:
    """Manage a single Chromium page and expose streaming + control hooks."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.browser = None
        self.page = None
        self.viewport: Dict[str, int] = DEFAULT_VIEWPORT.copy()
        self.frame_queue: queue.Queue[str] = queue.Queue(maxsize=4)
        self.initialized = threading.Event()
        self.started = False
        self._lock = threading.Lock()

    def ensure_started(self) -> None:
        with self._lock:
            if self.started:
                return
            logger.info("Starting Chromium controller thread.")
            self.thread.start()
            if not self.initialized.wait(timeout=30):
                raise RuntimeError("Chromium 初期化に失敗しました。")
            if self.page is None:
                raise RuntimeError("Chromium の起動中に問題が発生しました。")
            self.started = True

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._initialize())
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to initialize Chromium: %s", exc)
        finally:
            self.initialized.set()
        self.loop.run_forever()

    async def _initialize(self) -> None:
        self.browser = await launch(
            headless=True,
            handleSIGINT=False,
            handleSIGTERM=False,
            handleSIGHUP=False,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--hide-scrollbars",
            ],
        )
        self.page = await self.browser.newPage()
        await self.page.setViewport(self.viewport)
        await self.page.goto(DEFAULT_URL, {"waitUntil": "domcontentloaded"})
        client = self.page._client  # type: ignore[attr-defined]
        client.on("Page.screencastFrame", self._on_screencast_frame)
        await client.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": 70, "everyNthFrame": 1},
        )
        logger.info("Chromium ready and screencast started.")

    async def _on_screencast_frame(self, frame: Dict[str, Any]) -> None:
        data = frame.get("data")
        session_id = frame.get("sessionId")
        if data:
            try:
                self.frame_queue.put_nowait(data)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self.frame_queue.put_nowait(data)
        if session_id is not None and self.page is not None:
            await self.page._client.send(  # type: ignore[attr-defined]
                "Page.screencastFrameAck",
                {"sessionId": session_id},
            )

    def _run_coro(self, coro: Awaitable[Any]) -> None:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Chromium command failed: %s", exc)
            raise RuntimeError("Chromium への操作が失敗しました。") from exc

    def navigate(self, raw_url: str) -> None:
        if not raw_url:
            return
        url = normalize_url(raw_url)
        self._run_coro(self._navigate(url))

    async def _navigate(self, url: str) -> None:
        if self.page is None:
            return
        await self.page.goto(url, {"waitUntil": "networkidle2", "timeout": 45000})

    def mouse_event(self, payload: Dict[str, Any]) -> None:
        self._run_coro(self._handle_mouse_event(payload))

    async def _handle_mouse_event(self, payload: Dict[str, Any]) -> None:
        if self.page is None:
            return

        etype = payload.get("type")
        normalized_x = payload.get("x")
        normalized_y = payload.get("y")
        button = BUTTON_MAP.get(payload.get("button", 0), "left")

        def clamp(value: float | None) -> float | None:
            if value is None:
                return None
            numeric = float(value)
            if math.isnan(numeric):
                return None
            numeric = max(0.0, min(1.0, numeric))
            return numeric

        normalized_x = clamp(normalized_x)
        normalized_y = clamp(normalized_y)

        if normalized_x is not None and normalized_y is not None:
            abs_x = normalized_x * self.viewport["width"]
            abs_y = normalized_y * self.viewport["height"]
            if etype == "move":
                await self.page.mouse.move(abs_x, abs_y)
            elif etype == "down":
                await self.page.mouse.move(abs_x, abs_y)
                await self.page.mouse.down(button=button)
            elif etype == "up":
                await self.page.mouse.move(abs_x, abs_y)
                await self.page.mouse.up(button=button)
            elif etype == "click":
                await self.page.mouse.click(abs_x, abs_y, button=button)

        if etype == "wheel":
            delta_x = payload.get("deltaX", 0)
            delta_y = payload.get("deltaY", 0)
            await self.page.mouse.wheel(deltaX=float(delta_x), deltaY=float(delta_y))

    def keyboard_event(self, payload: Dict[str, Any]) -> None:
        self._run_coro(self._handle_keyboard_event(payload))

    async def _handle_keyboard_event(self, payload: Dict[str, Any]) -> None:
        if self.page is None:
            return
        etype = payload.get("type")
        key = payload.get("key")
        if not key or key == "Unidentified":
            return
        if etype == "keydown":
            await self.page.keyboard.down(key)
        elif etype == "keyup":
            await self.page.keyboard.up(key)

    def resize(self, width: int, height: int) -> None:
        width = max(320, min(1920, int(width)))
        height = max(240, min(1080, int(height)))
        self.viewport.update({"width": width, "height": height})
        self._run_coro(self._apply_viewport())

    async def _apply_viewport(self) -> None:
        if self.page is None:
            return
        await self.page.setViewport(self.viewport)


browser = BrowserController()
stream_task_started = threading.Event()


def ensure_streaming_background_task() -> None:
    if stream_task_started.is_set():
        return
    stream_task_started.set()

    def worker() -> None:
        while True:
            data = browser.frame_queue.get()
            socketio.emit("frame", {"data": data})

    socketio.start_background_task(worker)


@app.route("/")
def index() -> str:
    return render_template("index.html")


@socketio.on("connect")
def handle_connect() -> None:
    browser.ensure_started()
    ensure_streaming_background_task()
    emit("status", {"type": "info", "message": "Chromiumに接続しました。"})


@socketio.on("mouse-event")
def handle_mouse(payload: Dict[str, Any]) -> None:
    try:
        browser.mouse_event(payload or {})
    except RuntimeError as exc:
        emit("status", {"type": "error", "message": str(exc)})


@socketio.on("keyboard-event")
def handle_keyboard(payload: Dict[str, Any]) -> None:
    try:
        browser.keyboard_event(payload or {})
    except RuntimeError as exc:
        emit("status", {"type": "error", "message": str(exc)})


@socketio.on("navigate")
def handle_navigate(payload: Dict[str, Any]) -> None:
    url = (payload or {}).get("url", "")
    try:
        browser.navigate(url)
        emit("status", {"type": "info", "message": f"移動: {url}"})
    except RuntimeError as exc:
        emit("status", {"type": "error", "message": str(exc)})


@socketio.on("viewport")
def handle_viewport(payload: Dict[str, Any]) -> None:
    width = (payload or {}).get("width")
    height = (payload or {}).get("height")
    if width and height:
        browser.resize(int(width), int(height))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))
    logger.info("Starting Flask-SocketIO server on port %s.", port)
    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=True,
        allow_unsafe_werkzeug=True,
    )
