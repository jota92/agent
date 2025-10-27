from flask import Flask, Response, render_template_string, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException, NoAlertPresentException
from selenium.webdriver.common.by import By
import threading
import time
import io
import json
import base64
import os
import re
import hashlib
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from enum import Enum

app = Flask(__name__)
driver = None
lock = threading.Lock()
current_frame = None
running = True
frame_size = (1280, 800)
last_stream_request = 0.0

# AI Configuration
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
VLM_MAX_NEW_TOKENS = 384
AI_MAX_STEPS = 30
AI_STEP_TIMEOUT = 15

# Performance tuning
JPEG_QUALITY = 65
TARGET_STREAM_WIDTH = 1024
ACTIVE_CAPTURE_DELAY = 0.08
IDLE_CAPTURE_DELAY = 0.65
IDLE_THRESHOLD_SECONDS = 2.0
RESAMPLE_FILTER = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.LANCZOS)

class TaskStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class AIAction:
    action: str
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    reasoning: Optional[str] = None
    url: Optional[str] = None
    plan_step: Optional[int] = None

@dataclass
class AITask:
    task_id: str
    instruction: str
    status: TaskStatus
    current_step: int = 0
    max_steps: int = AI_MAX_STEPS
    logs: List[str] = None
    history: List[Dict] = None
    ai_cursor_pos: Optional[tuple] = None
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    plan_steps: List[str] = None
    plan_progress: int = 0
    stalled_frames: int = 0
    last_frame_hash: Optional[str] = None

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.history is None:
            self.history = []
        if self.plan_steps is None:
            self.plan_steps = []
        if self.created_at == 0:
            self.created_at = time.time()

ai_tasks: Dict[str, AITask] = {}
ai_task_lock = threading.Lock()
ai_cursor_position = None
ai_last_cursor_position = None

# Vision-language model state
vlm_model = None
vlm_processor = None
vlm_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vlm_load_lock = threading.Lock()
vlm_last_error: Optional[str] = None


def extract_search_query(instruction: str) -> Optional[str]:
    """Try to infer the keyword the user wants to search for."""
    if not instruction:
        return None
    search_space = instruction.strip()
    quote_patterns = [
        r'["“”](.+?)["“”]',
        r"['‘’](.+?)['‘’]",
        r'「(.+?)」',
        r'『(.+?)』'
    ]
    for pattern in quote_patterns:
        match = re.search(pattern, search_space)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate

    google_pattern = re.search(r'googleで(.+?)を?検索', instruction, re.IGNORECASE)
    if google_pattern:
        candidate = google_pattern.group(1).strip(" 。、,.!?\"'」』 ")
        if candidate:
            return candidate

    generic_pattern = re.search(
        r'(?:検索|search(?: for)?|look up|調べ(?:て)?|探して)\s*([^\s、。,。]+)',
        instruction,
        re.IGNORECASE
    )
    if generic_pattern:
        candidate = generic_pattern.group(1).strip(" 。、,.!?\"'」』 ")
        if candidate:
            return candidate
    return None


def build_task_plan(instruction: str) -> List[str]:
    """Create a lightweight, deterministic action plan from the natural instruction."""
    plan: List[str] = []
    normalized = instruction.lower()
    search_query = extract_search_query(instruction) or "the requested keyword"

    if "google" in normalized or "検索" in normalized:
        plan.append("Open https://www.google.com so the Google homepage is visible.")
        plan.append("Precisely click inside Google's search box to focus it.")
        plan.append(f"Type \"{search_query}\" into the search field and press Enter to run the search.")
        plan.append("Wait for the result page to finish rendering, then click the very first organic result.")
        plan.append("Confirm the opened page satisfies the instruction before finishing.")
    else:
        plan.append("Inspect the current page layout and locate the target area.")
        plan.append(f"Execute the instruction: {instruction}")
        plan.append("Validate that the requested goal is achieved, handling any popups or errors.")

    plan.append("Stay alert for unexpected dialogs or errors and resolve them before continuing.")
    return plan


def format_plan_text(plan_steps: List[str], completed_steps: int) -> str:
    """Pretty-print the structured plan so the VLM can reference it."""
    if not plan_steps:
        return "Plan: No predefined micro-steps; act directly but stay vigilant."

    lines = ["Structured plan:"]
    for idx, step in enumerate(plan_steps, start=1):
        if idx <= completed_steps:
            status = "✅ DONE"
        elif idx == completed_steps + 1:
            status = "➡ CURRENT"
        else:
            status = "… NEXT"
        lines.append(f"{idx}. {step} ({status})")
    return "\n".join(lines)


def instruction_mentions_google(instruction: str) -> bool:
    return "google" in instruction.lower() or "検索" in instruction


def build_forced_type_action(task: AITask, label: str = "recovering search flow") -> AIAction:
    """Create a deterministic 'type + Enter' action to nudge progress when stuck."""
    query = extract_search_query(task.instruction) or "search"
    total_steps = len(task.plan_steps) or 4
    next_step = min(max(task.plan_progress + 1, 1), total_steps)
    reasoning = f'Step {next_step}/{total_steps}: typing "{query}" and pressing Enter ({label})'
    return AIAction(action="type", text=f"{query}\\n", reasoning=reasoning)


def find_element_center(selectors: List[str]) -> Optional[tuple]:
    """Locate an element via CSS selectors and return its viewport center coordinates."""
    with lock:
        if not driver:
            init_browser()
        if not driver:
            return None
        for selector in selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
            except Exception:
                continue
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", element)
                rect = driver.execute_script("""
                    const r = arguments[0].getBoundingClientRect();
                    return {x: r.left, y: r.top, width: r.width, height: r.height};
                """, element)
            except Exception:
                continue
            if rect and rect.get('width', 0) > 1 and rect.get('height', 0) > 1:
                center_x = int(rect['x'] + rect['width'] / 2)
                center_y = int(rect['y'] + rect['height'] / 2)
                return (center_x, center_y)
    return None


def auto_target_click(action: AIAction, task: AITask) -> bool:
    """Adjust click coordinates using DOM metadata when the model output is unreliable."""
    if action.action != "click":
        return False
    if not instruction_mentions_google(task.instruction):
        return False
    step_hint = action.plan_step or (task.plan_progress + 1)
    needs_help = (
        action.x is None or action.y is None or
        (action.x >= 0 and action.y >= 0 and action.x < 40 and action.y < 40)
    )
    if not needs_help:
        return False

    query = extract_search_query(task.instruction) or ""
    updated = False
    if step_hint == 2:
        coords = find_element_center([
            "input[name='q']",
            "textarea[name='q']",
            "form[action='/search'] input[type='text']",
        ])
        if coords:
            action.x, action.y = coords
            task.logs.append("🎯 Auto-targeted Google search field using DOM geometry")
            updated = True
    elif step_hint >= 4:
        coords = find_element_center([
            "div#search h3",
            "div#search a h3",
            "a h3.LC20lb"
        ])
        if coords:
            action.x, action.y = coords
            task.logs.append("🎯 Auto-targeted first Google result heading via DOM geometry")
            updated = True
    return updated


def ensure_text_focus(task: AITask):
    """Ensure the most relevant text input is focused before typing."""
    if not instruction_mentions_google(task.instruction):
        return
    selectors = [
        "input[name='q']",
        "textarea[name='q']",
        "form[action='/search'] input[type='text']",
    ]
    try:
        with lock:
            if not driver:
                init_browser()
            if not driver:
                return
            driver.execute_script(
                """
                const selectors = arguments[0];
                for (const sel of selectors) {
                    const el = document.querySelector(sel);
                    if (el) {
                        if (document.activeElement !== el) {
                            el.focus();
                            if (typeof el.select === 'function') {
                                el.select();
                            }
                        }
                        return;
                    }
                }
                """,
                selectors
            )
    except Exception as exc:
        print(f"Focus helper failed: {exc}")


def ensure_search_start(task: AITask):
    """Navigate to Google proactively when the instruction explicitly asks for a Google search."""
    global last_stream_request
    if not instruction_mentions_google(task.instruction):
        return
    try:
        opened = False
        with lock:
            if not driver:
                init_browser()
            if driver:
                current_url = driver.current_url
                if "google" not in current_url.lower():
                    driver.get("https://www.google.com")
                    last_stream_request = time.time()
                    opened = True
        if opened:
            task.logs.append("🔎 Auto-opened Google to start the search workflow")
    except Exception as exc:
        task.logs.append(f"⚠️ Failed to auto-open Google: {exc}")


def update_stall_state(task: AITask, screenshot_bytes: bytes) -> bool:
    """Track whether the visual state is changing; return True if the frame is stale."""
    if not screenshot_bytes:
        return False
    frame_hash = hashlib.md5(screenshot_bytes).hexdigest()
    if task.last_frame_hash == frame_hash:
        task.stalled_frames += 1
    else:
        task.stalled_frames = 0
        task.last_frame_hash = frame_hash
    return task.stalled_frames >= 3


def update_plan_progress(task: AITask, action: AIAction):
    """Update plan progress using explicit plan_step or reasoning fallback."""
    if not task or not action:
        return
    with ai_task_lock:
        max_step = len(task.plan_steps)
        previous = task.plan_progress
        target = previous

        if action.action == "done":
            target = max_step
        elif action.plan_step:
            normalized = action.plan_step
            if max_step:
                normalized = min(action.plan_step, max_step)
            normalized = max(0, normalized)
            target = max(previous, normalized)
        elif action.reasoning and task.plan_steps:
            match = re.search(r'step\s*(\d+)', action.reasoning, re.IGNORECASE)
            if match:
                try:
                    inferred = int(match.group(1))
                except ValueError:
                    inferred = 0
                if inferred > 0:
                    normalized = min(inferred, max_step) if max_step else inferred
                    target = max(previous, normalized)

        if target > previous:
            task.plan_progress = target
            if task.plan_steps and target <= len(task.plan_steps):
                task.logs.append(f"🧭 Completed plan step {target}: {task.plan_steps[target-1]}")


def build_verification_question(task: AITask) -> Optional[str]:
    query = extract_search_query(task.instruction) or "the requested topic"
    if task.plan_progress == 3:
        return f"Is the Google search results page visible with results for \"{query}\" instead of the blank homepage?"
    if task.plan_progress == 4:
        return f"Has the first Google search result for \"{query}\" been opened or at least highlighted so it is clearly active?"
    if task.plan_progress >= 5:
        return f"Does the current page appear to belong to the requested destination for \"{query}\" (for example openai.com) rather than Google results?"
    return None


def call_vlm_verification(image_b64: str, question: str) -> Optional[bool]:
    try:
        load_vlm()
    except Exception as exc:
        print(f"Verification model unavailable: {exc}")
        return None
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as decode_error:
        print(f"Verification decode error: {decode_error}")
        return None

    prompt = f"""Question: {question}

Answer with only 'yes' or 'no'. Do not add explanations."""

    try:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]}
        ]
        chat_prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_processor(text=chat_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(vlm_device) for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.0,
                top_p=0.9,
                do_sample=False,
                eos_token_id=vlm_processor.tokenizer.eos_token_id,
                pad_token_id=vlm_processor.tokenizer.pad_token_id
            )
        generated_tokens = generated_ids[:, inputs['input_ids'].shape[1]:]
        generated_text = vlm_processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip().lower()
        print(f"🔍 Verification response: {generated_text}")
        if generated_text.startswith("yes"):
            return True
        if generated_text.startswith("no"):
            return False
    except Exception as exc:
        print(f"Verification inference error: {exc}")
        return None
    finally:
        image.close()
    return None


def verify_visual_progress(task: AITask):
    question = build_verification_question(task)
    if not question:
        return
    try:
        with lock:
            if not driver:
                init_browser()
            if not driver:
                return
            screenshot = driver.get_screenshot_as_png()
    except Exception as exc:
        task.logs.append(f"⚠️ Failed to capture screenshot for verification: {exc}")
        return
    image_b64 = base64.b64encode(screenshot).decode('utf-8')
    result = call_vlm_verification(image_b64, question)
    if result is True:
        task.logs.append(f"👁️ Visual verification passed: {question}")
    elif result is False:
        task.logs.append(f"⚠️ Visual verification failed: {question}")
        with ai_task_lock:
            task.plan_progress = max(0, task.plan_progress - 1)
    else:
        task.logs.append("⚠️ Visual verification inconclusive")


def wait_for_dom_idle(task: AITask, label: str = "", timeout: float = 6.0, poll_interval: float = 0.3) -> bool:
    """Wait until document.readyState settles to 'complete' twice in a row."""
    if not driver:
        return False
    end_time = time.time() + timeout
    consecutive_complete = 0
    while time.time() < end_time:
        try:
            with lock:
                if not driver:
                    init_browser()
                if not driver:
                    return False
                state = driver.execute_script("return document.readyState")
        except WebDriverException as exc:
            task.logs.append(f"⚠️ Failed to read DOM state ({label}): {exc}")
            return False
        if state == "complete":
            consecutive_complete += 1
            if consecutive_complete >= 2:
                return True
        else:
            consecutive_complete = 0
        time.sleep(poll_interval)
    task.logs.append(f"⏳ Page still loading after {label or 'action'}, proceeding cautiously")
    return False


def dismiss_js_alerts(task: AITask) -> bool:
    """Close blocking JavaScript alerts, confirmations, or prompts if present."""
    if not driver:
        return False
    try:
        with lock:
            if not driver:
                return False
            try:
                alert = driver.switch_to.alert
            except NoAlertPresentException:
                return False
            except Exception as exc:
                task.logs.append(f"⚠️ Alert detection failed: {exc}")
                return False
            text = (alert.text or "").strip()
            alert.accept()
    except Exception as exc:
        task.logs.append(f"⚠️ Failed to dismiss alert: {exc}")
        return False
    task.logs.append(f"🛡️ Auto-dismissed alert: {text[:80]}")
    time.sleep(0.3)
    return True


def load_vlm() -> None:
    """Load the local vision-language model if it has not been initialised yet."""
    global vlm_model, vlm_processor, vlm_last_error
    if vlm_model is not None and vlm_processor is not None:
        return
    with vlm_load_lock:
        if vlm_model is not None and vlm_processor is not None:
            return
        try:
            print(f"Loading vision-language model '{VLM_MODEL_ID}' on {vlm_device} ...")
            dtype = torch.float16 if vlm_device.type == 'cuda' else torch.float32
            processor = AutoProcessor.from_pretrained(VLM_MODEL_ID, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                VLM_MODEL_ID,
                dtype=dtype,
                trust_remote_code=True
            )
            model.to(vlm_device)
            model.eval()
            vlm_processor = processor
            vlm_model = model
            vlm_last_error = None
            print("Vision-language model ready.")
        except Exception as exc:
            vlm_model = None
            vlm_processor = None
            vlm_last_error = str(exc)
            print(f"Failed to load vision-language model: {exc}")
            raise

def init_browser():
    global driver, last_stream_request
    from selenium.webdriver.chrome.service import Service
    if driver:
        return driver
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--window-size=1280,800')
    chromedriver_path = os.environ.get('CHROMEDRIVER_PATH')
    service = None
    if chromedriver_path and os.path.isfile(chromedriver_path):
        service = Service(chromedriver_path)
    elif os.path.isfile('/usr/bin/chromedriver'):
        service = Service('/usr/bin/chromedriver')
    try:
        if service:
            driver_instance = webdriver.Chrome(service=service, options=options)
        else:
            driver_instance = webdriver.Chrome(options=options)
        driver_instance.get('https://www.google.com')
        driver = driver_instance
        last_stream_request = time.time()
        print("Chrome started")
    except Exception as exc:
        driver = None
        print(f"Browser init failed: {exc}")
        raise
    return driver

def restart_browser_locked(reason: str):
    global driver, current_frame
    print(f"Restarting browser: {reason}")
    if driver:
        try:
            driver.quit()
        except Exception as quit_error:
            print(f"Browser quit error: {quit_error}")
    driver = None
    current_frame = None
    try:
        init_browser()
    except Exception as init_error:
        print(f"Browser restart failed: {init_error}")

def draw_coordinate_grid(img: Image.Image) -> Image.Image:
    """座標グリッドを描画してAIが座標を見つけやすくする"""
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')
    width, height = img.size

    major_spacing = 200
    minor_spacing = 50
    tile_spacing = 100

    # 交互のタイルシェーディング（100px毎）
    for ty in range(0, height, tile_spacing):
        for tx in range(0, width, tile_spacing):
            tile_index = ((tx // tile_spacing) + (ty // tile_spacing)) % 2
            if tile_index == 0:
                draw.rectangle(
                    [(tx, ty), (min(tx + tile_spacing, width), min(ty + tile_spacing, height))],
                    fill=(255, 255, 255, 24)
                )

    # minor grid
    for x in range(0, width, minor_spacing):
        line_color = (255, 255, 255, 40) if x % major_spacing else (255, 255, 255, 120)
        line_width = 1 if x % major_spacing else 2
        draw.line([(x, 0), (x, height)], fill=line_color, width=line_width)

    for y in range(0, height, minor_spacing):
        line_color = (255, 255, 255, 40) if y % major_spacing else (255, 255, 255, 120)
        line_width = 1 if y % major_spacing else 2
        draw.line([(0, y), (width, y)], fill=line_color, width=line_width)

    # Axis ribbons
    header_height = 28
    sidebar_width = 80
    draw.rectangle([(0, 0), (width, header_height)], fill=(0, 0, 0, 160))
    draw.rectangle([(0, 0), (sidebar_width, height)], fill=(0, 0, 0, 150))

    # Column labels (every 200px)
    for x in range(0, width + 1, major_spacing):
        label = f"{x}px"
        text_pos = (max(sidebar_width + 4, x + 4) if x == 0 else x + 4, 4)
        draw.text(text_pos, label, fill=(255, 255, 0, 220))
        if x > 0:
            draw.text((x + 4, header_height + 4), label, fill=(210, 255, 140, 220))

    # Row labels (every 200px)
    for y in range(0, height + 1, major_spacing):
        label = f"{y}px"
        text_y = y + 4 if y + 20 < height else height - 24
        draw.text((4, max(header_height + 4, text_y)), label, fill=(255, 255, 0, 220))
        if y > 0:
            draw.text((sidebar_width + 6, y + 4), label, fill=(210, 255, 140, 220))

    # Quadrant markers
    letter_labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, x in enumerate(range(0, width, tile_spacing)):
        letter = letter_labels[idx % len(letter_labels)]
        draw.text((x + 4, header_height - 20), letter, fill=(0, 200, 255, 220))

    for idx, y in enumerate(range(0, height, tile_spacing)):
        draw.text((sidebar_width - 24, y + 4), f"{idx:02d}", fill=(0, 200, 255, 220))

    # Center crosshair
    center_x, center_y = width // 2, height // 2
    draw.line([(center_x - 30, center_y), (center_x + 30, center_y)], fill=(0, 255, 0, 220), width=3)
    draw.line([(center_x, center_y - 30), (center_x, center_y + 30)], fill=(0, 255, 0, 220), width=3)
    draw.ellipse([(center_x - 6, center_y - 6), (center_x + 6, center_y + 6)], outline=(0, 255, 0, 220), width=2)
    draw.text((center_x + 34, center_y - 12), f"({center_x}, {center_y})", fill=(0, 255, 0, 230))

    composed = Image.alpha_composite(base, overlay)
    return composed.convert(img.mode)

def draw_ai_cursor(img: Image.Image, pos: tuple) -> Image.Image:
    """Mac風のポインターを強調表示する"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    x, y = pos

    # グロー効果
    glow_radius = 55
    glow_color = (66, 133, 244, 80)
    draw.ellipse([(x - glow_radius, y - glow_radius), (x + glow_radius, y + glow_radius)], fill=glow_color)

    # ポインタ本体（三角形）
    pointer_height = 46
    pointer_width = 28
    pointer = [
        (int(x), int(y)),
        (int(x), int(y + pointer_height)),
        (int(x + pointer_width), int(y + pointer_height * 0.55))
    ]

    # 影
    shadow_offset = 4
    shadow = [(px + shadow_offset, py + shadow_offset) for px, py in pointer]
    draw.polygon(shadow, fill=(0, 0, 0, 160))

    # 白いポインタ
    draw.polygon(pointer, fill=(255, 255, 255, 255))
    draw.line(pointer + [pointer[0]], fill=(50, 50, 50, 255), width=2)

    # テールの小さな四角（Mac風の持ち手）
    tail_width = 10
    tail_height = 14
    tail_origin_x = int(x + pointer_width * 0.55)
    tail_origin_y = int(y + pointer_height * 0.55)
    tail_box = [
        tail_origin_x,
        tail_origin_y,
        tail_origin_x + tail_width,
        tail_origin_y + tail_height
    ]
    draw.rectangle(tail_box, fill=(180, 180, 180, 255), outline=(50, 50, 50, 255))

    return img_copy

def capture_loop():
    global current_frame, running, frame_size, last_stream_request, ai_cursor_position, ai_last_cursor_position
    while running:
        try:
            now = time.time()
            if now - last_stream_request > IDLE_THRESHOLD_SECONDS:
                time.sleep(IDLE_CAPTURE_DELAY)
                continue
            png = None
            with lock:
                if not driver:
                    try:
                        init_browser()
                    except Exception as init_error:
                        print(f"Capture init retry failed: {init_error}")
                        time.sleep(2)
                        continue
                if driver:
                    try:
                        png = driver.get_screenshot_as_png()
                    except WebDriverException as grab_error:
                        restart_browser_locked(f"screenshot failure: {grab_error}")
                        time.sleep(0.5)
                        continue
            if png:
                image = Image.open(io.BytesIO(png))
                frame_size = image.size
                
                # まず座標グリッドを描画
                processed = draw_coordinate_grid(image)
                
                # AIカーソルを元の解像度で描画（縮小前に必ず描画）
                with ai_task_lock:
                    cursor_pos = ai_cursor_position or ai_last_cursor_position
                
                if cursor_pos:
                    print(f"🎯 Drawing AI cursor at: {cursor_pos}")  # デバッグログ
                    processed = draw_ai_cursor(processed, cursor_pos)
                
                # その後に縮小処理
                if TARGET_STREAM_WIDTH and processed.width > TARGET_STREAM_WIDTH:
                    ratio = TARGET_STREAM_WIDTH / float(processed.width)
                    new_height = max(1, int(processed.height * ratio))
                    resized = processed.resize((TARGET_STREAM_WIDTH, new_height), RESAMPLE_FILTER)
                    if processed is not image:
                        processed.close()
                    processed = resized
                
                jpeg_buffer = io.BytesIO()
                processed.convert('RGB').save(jpeg_buffer, format='JPEG', quality=JPEG_QUALITY)
                with lock:
                    current_frame = jpeg_buffer.getvalue()
                if processed is not image:
                    processed.close()
                image.close()
            time.sleep(ACTIVE_CAPTURE_DELAY)
        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(1)

def call_vlm_action(
    image_b64: str,
    instruction: str,
    step_num: int,
    history: List[Dict] = None,
    plan: Optional[List[str]] = None,
    plan_progress: int = 0,
    stalled_frames: int = 0
) -> Optional[AIAction]:
    if history is None:
        history = []
    plan = plan or []
    plan_length = len(plan)
    plan_progress = max(0, min(plan_progress, plan_length))

    def _coerce_int(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(round(value))
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return int(round(float(value)))
            except ValueError:
                return None
        return None

    # 直近3ステップの履歴を整形
    history_lines: List[str] = []
    last_action_type = None
    if history:
        for h in history[-3:]:
            act_dict = h.get('action', {})
            action_name = act_dict.get('action', 'unknown')
            reasoning = act_dict.get('reasoning', 'N/A')
            x = act_dict.get('x', '')
            y = act_dict.get('y', '')
            coords = f" at ({x},{y})" if x and y else ""
            history_lines.append(f"• Step {h['step']}: {action_name.upper()}{coords} - {reasoning}")

        last_action_type = history[-1].get('action', {}).get('action')
    else:
        history_lines.append("No previous actions. Start with the first plan step.")

    if last_action_type == "click":
        history_lines.append("⚠️ Previous action was a CLICK. Focus a new element (type/wait) before clicking again.")
    elif last_action_type == "type":
        history_lines.append("⚠️ Previous action was TYPE. Pause to verify the typed text before more typing.")
    elif last_action_type == "wait":
        history_lines.append("⚠️ Previous action was WAIT. Re-check the page and proceed deliberately.")

    plan_text = format_plan_text(plan, plan_progress)
    screen_status = (
        f"⚠️ Screen alert: View unchanged for {stalled_frames} consecutive captures. Confirm an update or wait/scroll before repeating."
        if stalled_frames >= 3 else
        "Screen status: Visual updates detected normally."
    )

    history_text = "\n".join(history_lines)

    prompt = f"""Task: {instruction}

{plan_text}

Current agent step: {step_num}
Recent context:
{history_text}

Screen details:
- Screenshot already includes a yellow coordinate grid and the current cursor.
- Coordinates are absolute pixels (x, y) from the grid overlay.
- Alternating 100px tiles are lettered along the top (A, B, C, …) and numbered down the left (00, 01, 02 …); minor lines every 50px and bold lines every 200px.
- Think carefully before selecting an action; avoid repeating the exact same coordinates.
- {screen_status}

Allowed actions and required fields:
- click → requires integer x, y, and reasoning.
- type → requires text (include \\n for Enter when needed) and reasoning.
- wait → requires reasoning.
- scroll → requires reasoning.
- done → requires reasoning only when task is achieved.
- plan_step → REQUIRED integer referencing which plan item (1-based) you are executing right now.

Critical rules:
- Break the goal into the listed micro-steps and complete them sequentially.
- If the instruction mentions Google search, always: open Google → click the search box → type the query → press Enter → click the top result.
- Monitor the page for popups, dialogs, or unexpected screens and handle them before proceeding.
- Before typing, ensure the correct input is focused (click first if needed). After typing a query that must be submitted, append \\n to press Enter.
- If something looks wrong (different page, missing button), take corrective actions such as wait, scroll, or re-focus instead of repeating the same click.
- Begin the reasoning text with "Step X/Y:" to identify the plan item you are executing and keep plan_step aligned with that number.

STRICT OUTPUT FORMAT:
Return exactly ONE JSON object following this schema:
{{
    "action": "click" | "type" | "wait" | "scroll" | "done",
    "reasoning": "short explanation",
    "x": <int> (only for click),
    "y": <int> (only for click),
    "text": "..." (only for type),
    "plan_step": <int>
}}
Do NOT write anything outside the JSON object. No Markdown, no lists, no explanations. Only the JSON object."""

    try:
        load_vlm()
    except Exception as load_error:
        print(f"❌ Vision model unavailable: {load_error}")
        return None

    generation_options = {"temperature": 0.0, "top_p": 0.7, "top_k": 10}

    image = None
    try:
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as decode_error:
        print(f"❌ Failed to decode screenshot for VLM: {decode_error}")
        return None

    try:
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a precise web automation agent. Follow the task, reason about the grid coordinates, and respond with a single JSON object describing the next action. Never include any extra words."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        chat_prompt = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = vlm_processor(text=chat_prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to(vlm_device) for k, v in inputs.items()}

        do_sample = generation_options["temperature"] > 0.0

        with torch.no_grad():
            generated_ids = vlm_model.generate(
                **inputs,
                max_new_tokens=VLM_MAX_NEW_TOKENS,
                temperature=generation_options["temperature"],
                top_p=generation_options["top_p"],
                top_k=generation_options["top_k"],
                do_sample=do_sample,
                eos_token_id=vlm_processor.tokenizer.eos_token_id,
                pad_token_id=vlm_processor.tokenizer.pad_token_id
            )

        generated_tokens = generated_ids[:, inputs['input_ids'].shape[1]:]
        generated_text = vlm_processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        print(f"🤖 AI Response: {generated_text[:300]}")

        if '{' in generated_text and '}' in generated_text:
            json_start = generated_text.find('{')
            decoder = json.JSONDecoder()
            try:
                action_data, index = decoder.raw_decode(generated_text[json_start:])
                trailing = generated_text[json_start + index:].strip()
                if trailing:
                    print(f"ℹ️ Ignored trailing content after JSON: {trailing[:200]}")
                for numeric_key in ("x", "y", "plan_step"):
                    if numeric_key in action_data:
                        action_data[numeric_key] = _coerce_int(action_data[numeric_key])
                print(f"✓ Parsed JSON: {action_data}")
                return AIAction(**action_data)
            except json.JSONDecodeError as decode_json_error:
                candidate = generated_text[json_start:]
                print(f"JSON parse error: {decode_json_error}")
                print(f"Attempted to parse (truncated): {candidate[:300]}")
                return None
        else:
            print(f"❌ No JSON braces found in response: {generated_text[:300]}")
            return None
    except Exception as inference_error:
        print(f"❌ Vision model inference error: {inference_error}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        if image:
            image.close()

def execute_ai_action(action: AIAction, task: AITask) -> bool:
    global ai_cursor_position, ai_last_cursor_position, last_stream_request
    try:
        dismiss_js_alerts(task)

        if action.action == "click":
            auto_target_click(action, task)
            if action.x is None or action.y is None:
                task.logs.append("⚠️ Click skipped due to missing coordinates even after auto-targeting")
                return True
            with ai_task_lock:
                start_pos = ai_cursor_position or ai_last_cursor_position
                if start_pos is None:
                    start_pos = (frame_size[0] // 2, frame_size[1] // 2)
            bounded_x = max(0, min(frame_size[0] - 1, int(action.x)))
            bounded_y = max(0, min(frame_size[1] - 1, int(action.y)))
            action.x = bounded_x
            action.y = bounded_y
            end_pos = (bounded_x, bounded_y)

            animation_steps = 20
            animation_delay = 0.03
            for step in range(animation_steps + 1):
                progress = step / animation_steps
                eased_progress = progress * progress * (3 - 2 * progress)
                current_x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * eased_progress)
                current_y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * eased_progress)
                with ai_task_lock:
                    ai_cursor_position = (current_x, current_y)
                    ai_last_cursor_position = ai_cursor_position
                    task.ai_cursor_pos = ai_cursor_position
                time.sleep(animation_delay)

            time.sleep(0.25)

            with lock:
                if not driver:
                    init_browser()
                if not driver:
                    raise RuntimeError("Browser driver unavailable for click action")
                last_stream_request = time.time()
                metrics = driver.execute_cdp_cmd('Page.getLayoutMetrics', {})
                layout = metrics.get('layoutViewport', {})
                visual = metrics.get('visualViewport', {})
                layout_width = visual.get('clientWidth') or layout.get('clientWidth', frame_size[0])
                layout_height = visual.get('clientHeight') or layout.get('clientHeight', frame_size[1])
                scale_x = layout_width / frame_size[0] if frame_size[0] else 1
                scale_y = layout_height / frame_size[1] if frame_size[1] else 1
                offset_x = visual.get('pageX', 0)
                offset_y = visual.get('pageY', 0)
                target_x = action.x * scale_x + offset_x
                target_y = action.y * scale_y + offset_y
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
                    'type': 'mouseMoved',
                    'x': target_x,
                    'y': target_y,
                    'button': 'none'
                })
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
                    'type': 'mousePressed',
                    'x': target_x,
                    'y': target_y,
                    'button': 'left',
                    'clickCount': 1
                })
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
                    'type': 'mouseReleased',
                    'x': target_x,
                    'y': target_y,
                    'button': 'left',
                    'clickCount': 1
                })

            task.logs.append(f"✓ Clicked at ({action.x}, {action.y}): {action.reasoning}")
            time.sleep(0.4)
            wait_for_dom_idle(task, "click")
            with ai_task_lock:
                ai_cursor_position = end_pos
                ai_last_cursor_position = end_pos
                task.ai_cursor_pos = end_pos

        elif action.action == "type" and action.text:
            ensure_text_focus(task)
            normalized_text = action.text.replace("\\n", "\n")
            segments = normalized_text.split("\n")

            with lock:
                if not driver:
                    init_browser()
                if not driver:
                    raise RuntimeError("Browser driver unavailable for typing")
                last_stream_request = time.time()

                for idx, segment in enumerate(segments):
                    if segment:
                        driver.execute_cdp_cmd('Input.insertText', {'text': segment})
                        time.sleep(0.05)
                    if idx < len(segments) - 1:
                        driver.execute_cdp_cmd('Input.dispatchKeyEvent', {
                            'type': 'rawKeyDown',
                            'key': 'Enter',
                            'code': 'Enter',
                            'windowsVirtualKeyCode': 13,
                            'nativeVirtualKeyCode': 13
                        })
                        driver.execute_cdp_cmd('Input.dispatchKeyEvent', {
                            'type': 'keyUp',
                            'key': 'Enter',
                            'code': 'Enter',
                            'windowsVirtualKeyCode': 13,
                            'nativeVirtualKeyCode': 13
                        })
                        time.sleep(0.05)

            enter_count = max(0, len(segments) - 1)
            typed_preview = normalized_text.replace("\n", "\\n")
            if enter_count:
                task.logs.append(f"✓ Typed: '{typed_preview}' + ENTER x{enter_count} - {action.reasoning}")
            else:
                task.logs.append(f"✓ Typed: '{typed_preview}' - {action.reasoning}")
            wait_for_dom_idle(task, "type")
            with ai_task_lock:
                if ai_cursor_position:
                    ai_last_cursor_position = ai_cursor_position
                elif ai_last_cursor_position:
                    ai_cursor_position = ai_last_cursor_position
                task.ai_cursor_pos = ai_cursor_position or ai_last_cursor_position

        elif action.action == "navigate" and action.url:
            with lock:
                if not driver:
                    init_browser()
                if not driver:
                    raise RuntimeError("Browser driver unavailable for navigation")
                last_stream_request = time.time()
                driver.get(action.url)
            task.logs.append(f"✓ Navigated to: {action.url}")
            wait_for_dom_idle(task, "navigate")
            with ai_task_lock:
                if ai_cursor_position:
                    ai_last_cursor_position = ai_cursor_position
                task.ai_cursor_pos = ai_cursor_position or ai_last_cursor_position

        elif action.action == "scroll":
            delta = 300
            if action.text:
                text_lower = action.text.lower()
                if any(word in text_lower for word in ["up", "上"]):
                    delta = -300
                match = re.search(r'(-?\d+)', action.text)
                if match:
                    try:
                        magnitude = int(match.group(1))
                        if magnitude != 0:
                            delta = magnitude
                    except ValueError:
                        pass
            with lock:
                if not driver:
                    init_browser()
                if not driver:
                    raise RuntimeError("Browser driver unavailable for scroll")
                last_stream_request = time.time()
                driver.execute_script("window.scrollBy(0, arguments[0])", delta)
            direction = "up" if delta < 0 else "down"
            task.logs.append(f"✓ Scrolled {direction} ({delta}px) - {action.reasoning}")
            with ai_task_lock:
                if ai_cursor_position:
                    ai_last_cursor_position = ai_cursor_position
                task.ai_cursor_pos = ai_cursor_position or ai_last_cursor_position

        elif action.action == "wait":
            time.sleep(2)
            task.logs.append(f"⏳ Waiting - {action.reasoning}")
            wait_for_dom_idle(task, "wait")
            with ai_task_lock:
                if ai_cursor_position:
                    ai_last_cursor_position = ai_cursor_position
                task.ai_cursor_pos = ai_cursor_position or ai_last_cursor_position

        elif action.action == "done":
            task.logs.append(f"✅ Task completed - {action.reasoning}")
            with ai_task_lock:
                ai_cursor_position = None
                ai_last_cursor_position = None
                task.ai_cursor_pos = None
            return False

        time.sleep(0.5)
        return True
    except Exception as e:
        task.logs.append(f"Action failed: {e}")
        print(f"Execute action error: {e}")
        return True

def ai_agent_loop(task_id: str):
    global ai_cursor_position, ai_last_cursor_position
    with ai_task_lock:
        if task_id not in ai_tasks:
            return
        task = ai_tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
    print(f"AI Agent starting task: {task.instruction}")
    ensure_search_start(task)
    
    last_action_signature = None  # 前回のアクション記録
    consecutive_click_count = 0  # 連続クリック回数
    last_click_pos = None  # 前回のクリック座標
    duplicate_signature_count = 0  # 同一アクション連続回数
    
    try:
        for step in range(task.max_steps):
            with ai_task_lock:
                if task.status != TaskStatus.RUNNING:
                    print(f"Task {task_id} stopped by user")
                    break
                task.current_step = step + 1
            with lock:
                if not driver:
                    init_browser()
                png = driver.get_screenshot_as_png()
            plan_snapshot: List[str] = []
            plan_progress_snapshot = 0
            stalled_frames_snapshot = 0
            should_force_refresh = False

            screenshot = Image.open(io.BytesIO(png)).convert('RGB')
            width, height = screenshot.size
            with ai_task_lock:
                stall_triggered = update_stall_state(task, png)
                stalled_frames_snapshot = task.stalled_frames
                plan_snapshot = list(task.plan_steps)
                plan_progress_snapshot = task.plan_progress
                thinking_cursor = ai_cursor_position or ai_last_cursor_position
                if stall_triggered and stalled_frames_snapshot == 3:
                    task.logs.append("⚠️ Screen unchanged for 3 captures. AI will attempt corrective action.")
                if stall_triggered and stalled_frames_snapshot >= 6:
                    task.logs.append("♻️ Screen frozen for 6+ captures. Forcing browser refresh.")
                    should_force_refresh = True
            if not thinking_cursor:
                thinking_cursor = (width // 2, height // 2)
            if should_force_refresh:
                screenshot.close()
                with lock:
                    try:
                        driver.refresh()
                    except Exception as refresh_error:
                        with ai_task_lock:
                            task.logs.append(f"⚠️ Refresh failed: {refresh_error}")
                time.sleep(2)
                continue

            annotated_frame = draw_coordinate_grid(screenshot)
            if thinking_cursor:
                annotated_with_cursor = draw_ai_cursor(annotated_frame, thinking_cursor)
                if annotated_with_cursor is not annotated_frame:
                    annotated_frame.close()
                annotated_frame = annotated_with_cursor

            buffered = io.BytesIO()
            annotated_frame.save(buffered, format="PNG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            buffered.close()
            annotated_frame.close()
            screenshot.close()
            print(f"Step {step + 1}: Asking AI for next action...")
            action = call_vlm_action(
                image_b64,
                task.instruction,
                step + 1,
                task.history,
                plan=plan_snapshot,
                plan_progress=plan_progress_snapshot,
                stalled_frames=stalled_frames_snapshot
            )
            
            time.sleep(0.1) # UI更新のための短い待機

            if not action:
                task.logs.append("AI failed to provide valid action")
                time.sleep(2)
                continue

            if action.action == "done" and task.plan_steps and task.plan_progress < len(task.plan_steps):
                task.logs.append("⚠️ Model attempted to finish before completing all plan steps. Re-evaluating.")
                time.sleep(1)
                continue
            
            # より厳格な重複防止ロジック
            # 1. 完全一致の重複チェック
            action_signature = f"{action.action}:{action.x}:{action.y}:{action.text}"
            if action_signature == last_action_signature:
                duplicate_signature_count += 1
                forced_action = None
                if action.action == "click" and duplicate_signature_count >= 2:
                    forced_action = build_forced_type_action(task, "auto-recovery after duplicate clicks")
                if forced_action:
                    task.logs.append("🔄 Forced type action due to repeated identical clicks")
                    action = forced_action
                    action_signature = f"{action.action}:{action.x}:{action.y}:{action.text}"
                    consecutive_click_count = 0
                    last_click_pos = None
                    duplicate_signature_count = 0
                else:
                    task.logs.append(f"⚠️ Exact duplicate detected, skipping: {action.action}")
                    print(f"⚠️ Prevented exact duplicate: {action_signature}")
                    time.sleep(1)
                    continue
            else:
                duplicate_signature_count = 0
            
            # 2. 近い座標へのクリックを重複とみなす
            if action.action == "click" and last_click_pos:
                distance = ((action.x - last_click_pos[0])**2 + (action.y - last_click_pos[1])**2)**0.5
                if distance < 100:  # 100ピクセル以内
                    task.logs.append(f"⚠️ Too close to last click ({int(distance)}px), skipping")
                    print(f"⚠️ Click too close to last position: {distance:.1f}px")
                    consecutive_click_count += 1
                    # 3回連続で近いクリックなら強制的にtypeに進む
                    if consecutive_click_count >= 2:
                        task.logs.append("🔄 Forcing type action to break loop")
                        print("🔄 Too many similar clicks, forcing type action")
                        # 検索クエリを抽出
                        match = re.search(r'["\'](.+?)["\'].*検索', task.instruction)
                        search_query = match.group(1) if match else "検索"
                        action = AIAction(action="type", text=f"{search_query}\\n", reasoning="Forced action to break click loop")
                        consecutive_click_count = 0
                        last_click_pos = None
                    else:
                        time.sleep(1)
                        continue
            
            # クリックの場合は座標を記録
            if action.action == "click":
                last_click_pos = (action.x, action.y)
                consecutive_click_count += 1
            else:
                consecutive_click_count = 0
                last_click_pos = None
            
            last_action_signature = action_signature
            
            task.history.append({'step': step + 1, 'action': asdict(action), 'timestamp': time.time()})
            print(f"Action: {action.action} - {action.reasoning}")
            should_continue = execute_ai_action(action, task)
            prior_progress = task.plan_progress
            update_plan_progress(task, action)
            if task.plan_progress != prior_progress:
                verify_visual_progress(task)
            if task.plan_steps and task.plan_progress >= len(task.plan_steps):
                task.logs.append("✅ All planned steps have been addressed")
            if not should_continue:
                with ai_task_lock:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    ai_cursor_position = None
                    ai_last_cursor_position = None
                    task.ai_cursor_pos = None
                print(f"Task {task_id} completed successfully")
                break
            time.sleep(1)
        else:
            with ai_task_lock:
                task.status = TaskStatus.FAILED
                task.error = "Maximum steps reached"
                task.completed_at = time.time()
                ai_cursor_position = None
                ai_last_cursor_position = None
                task.ai_cursor_pos = None
            print(f"Task {task_id} failed: max steps reached")
    except Exception as e:
        with ai_task_lock:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            ai_cursor_position = None
            ai_last_cursor_position = None
            task.ai_cursor_pos = None
        print(f"Task {task_id} failed: {e}")

@app.route('/')
def index():
    html = '''<!DOCTYPE html>
<html><head><title>AI Browser Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0a;height:100vh;display:flex;font-family:'Segoe UI',Arial,sans-serif;color:#fff}
.sidebar{width:380px;background:#1a1a1a;border-right:1px solid #333;display:flex;flex-direction:column;overflow:hidden}
.sidebar-header{padding:20px;background:#222;border-bottom:1px solid #333}
.sidebar-header h1{font-size:20px;margin-bottom:8px;color:#4a9eff}
.sidebar-header p{font-size:12px;color:#888}
.task-input{padding:16px;border-bottom:1px solid #333}
.task-input textarea{width:100%;padding:12px;background:#2a2a2a;border:1px solid #444;border-radius:8px;color:#fff;font-size:14px;resize:vertical;min-height:80px}
.task-input textarea:focus{outline:none;border-color:#4a9eff}
.task-controls{display:flex;gap:8px;margin-top:12px}
.task-controls button{flex:1;padding:10px;border:none;border-radius:6px;cursor:pointer;font-size:14px;font-weight:600;transition:all 0.2s}
.btn-primary{background:#4a9eff;color:#fff}
.btn-primary:hover{background:#3a8eef}
.btn-primary:disabled{background:#666;cursor:not-allowed}
.btn-danger{background:#ef4444;color:#fff}
.btn-danger:hover{background:#dc2626}
.btn-success{background:#10b981;color:#fff}
.btn-success:hover{background:#059669}
.task-status{padding:16px;border-bottom:1px solid #333;max-height:200px;overflow-y:auto}
.status-item{margin-bottom:8px;padding:8px;background:#2a2a2a;border-radius:4px;font-size:13px}
.status-running{border-left:3px solid #4a9eff}
.status-completed{border-left:3px solid #10b981}
.status-failed{border-left:3px solid #ef4444}
.status-stopped{border-left:3px solid #f59e0b}
.task-logs{flex:1;padding:16px;overflow-y:auto;font-family:monospace;font-size:12px}
.log-entry{padding:6px 0;border-bottom:1px solid #222;color:#ccc}
.log-entry.success{color:#10b981}
.log-entry.error{color:#ef4444}
.log-entry.info{color:#4a9eff}
.plan-panel{padding:16px;border-bottom:1px solid #333;background:#181818;max-height:220px;overflow-y:auto}
.plan-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.plan-header h2{font-size:13px;text-transform:uppercase;letter-spacing:0.8px;color:#bbb}
.plan-progress{font-size:12px;color:#eee;background:#262626;padding:2px 10px;border-radius:999px;border:1px solid #333}
.plan-steps{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:6px}
.plan-step{display:flex;gap:8px;font-size:12px;padding:6px;border-left:3px solid #333;background:#111;border-radius:4px;color:#ddd}
.plan-step .plan-step-index{font-weight:700;color:#777;min-width:16px;text-align:right}
.plan-step.done{border-color:#10b981;color:#9ae6b4;background:#0f2a1e;text-decoration:line-through;opacity:0.8}
.plan-step.current{border-color:#4a9eff;background:rgba(74,158,255,0.12);color:#f0f6ff;box-shadow:0 0 8px rgba(74,158,255,0.25)}
.plan-step.next{opacity:0.7}
.main{flex:1;display:flex;flex-direction:column;background:#000}
.browser-view{flex:1;display:flex;align-items:center;justify-content:center;position:relative;overflow:auto}
.browser-view img{max-width:100%;max-height:100%;cursor:crosshair}
.msg{position:absolute;top:20px;left:50%;transform:translateX(-50%);background:rgba(74,158,255,0.95);color:#fff;padding:12px 24px;border-radius:8px;font-size:14px;display:none;box-shadow:0 4px 12px rgba(0,0,0,0.3)}
.ollama-status{position:absolute;top:12px;right:12px;padding:6px 12px;background:rgba(26,26,26,0.9);border:1px solid #333;border-radius:6px;font-size:11px;display:flex;align-items:center;gap:6px}
.status-dot{width:8px;height:8px;border-radius:50%;background:#10b981}
.status-dot.disconnected{background:#ef4444}
.cursor-info{position:absolute;bottom:12px;left:12px;padding:10px 16px;background:rgba(26,26,26,0.95);border:2px solid #ff3232;border-radius:8px;font-size:13px;font-family:monospace;color:#ff6464;display:none;box-shadow:0 0 30px rgba(255,50,50,0.4);backdrop-filter:blur(10px)}
.cursor-info.active{display:block;animation:pulse 1.5s ease-in-out infinite}
.cursor-info .label{color:#888;margin-right:8px;font-weight:600}
.cursor-info .value{color:#ff6464;font-weight:700;font-size:14px}
@keyframes pulse{0%,100%{box-shadow:0 0 30px rgba(255,50,50,0.4)}50%{box-shadow:0 0 40px rgba(255,50,50,0.6)}}
.cursor-status{display:block;margin-top:4px;font-size:11px;color:#ffa500;font-weight:600}
</style></head><body>
<div class="sidebar">
<div class="sidebar-header">
<h1>🤖 AI Browser Agent</h1>
<p>Autonomous web navigation powered by local AI</p>
</div>
<div class="task-input">
<textarea id="taskInput" placeholder="例: Googleで'OpenAI'を検索して、最初の結果をクリックしてください"></textarea>
<div class="task-controls">
<button class="btn-primary" onclick="startTask()" id="startBtn">▶ Start Task</button>
<button class="btn-danger" onclick="stopTask()" id="stopBtn" style="display:none">⏹ Stop</button>
<button class="btn-success" onclick="retryTask()" id="retryBtn" style="display:none">🔄 Retry</button>
</div>
</div>
<div class="task-status" id="taskStatus">
<div class="status-item">No active tasks</div>
</div>
<div class="plan-panel" id="planPanel">
<div class="plan-header">
<h2>Agent Plan</h2>
<span class="plan-progress" id="planProgressLabel">0/0</span>
</div>
<ol class="plan-steps" id="planList">
<li class="plan-step next"><span class="plan-step-index">-</span><span>No plan initialized yet.</span></li>
</ol>
</div>
<div class="task-logs" id="taskLogs">
<div class="log-entry info">✓ Ready. Waiting for task...</div>
</div>
</div>
<div class="main">
<div class="browser-view">
<div class="msg" id="msg">Loading...</div>
<div class="ollama-status">
<div class="status-dot" id="ollamaDot"></div>
<span id="ollamaText">Checking local VLM...</span>
</div>
<div class="cursor-info" id="cursorInfo">
<span class="label">AI Cursor:</span>
<span id="cursorCoords">-</span>
</div>
<img id="browserImg">
</div>
</div>
<script>
let currentTaskId=null;
let lastInstruction='';
let fetchingStream=false;
let lastObjectUrl=null;

async function update(){
if(fetchingStream) return;
fetchingStream=true;
try{
const response=await fetch('/stream?'+Date.now(),{cache:'no-store'});
if(!response.ok) throw new Error('stream '+response.status);
const blob=await response.blob();
const objectUrl=URL.createObjectURL(blob);
if(lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
lastObjectUrl=objectUrl;
browserImg.src=objectUrl;
}catch(err){
console.log(err);
}finally{
fetchingStream=false;
}
}

async function checkOllama(){
try{
const response=await fetch('/ai/health');
const data=await response.json();
if(data.ollama_available){
ollamaDot.className='status-dot';
const deviceLabel=data.device?' @ '+data.device:'';
ollamaText.textContent='Local VLM: '+data.model+deviceLabel;
}else{
ollamaDot.className='status-dot disconnected';
ollamaText.textContent='Local VLM: Not ready';
}
}catch(e){
ollamaDot.className='status-dot disconnected';
ollamaText.textContent='Local VLM: Error';
}
}

async function startTask(){
const instruction=taskInput.value.trim();
if(!instruction){
alert('Please enter a task instruction');
return;
}
lastInstruction=instruction;
try{
const response=await fetch('/ai/task',{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({instruction})
});
const data=await response.json();
if(data.task_id){
currentTaskId=data.task_id;
startBtn.style.display='none';
stopBtn.style.display='block';
retryBtn.style.display='none';
startBtn.disabled=true;
msg.textContent='🤖 AI Agent working...';
msg.style.display='block';
addLog('info','Task started: '+instruction);
}
}catch(e){
alert('Failed to start task: '+e.message);
}
}

async function stopTask(){
if(!currentTaskId) return;
try{
await fetch('/ai/stop/'+currentTaskId,{method:'POST'});
currentTaskId=null;
startBtn.style.display='block';
stopBtn.style.display='none';
retryBtn.style.display='block';
startBtn.disabled=false;
msg.style.display='none';
cursorInfo.classList.remove('active');
addLog('info','⏹ Task stopped by user');
}catch(e){
console.error(e);
}
}

async function retryTask(){
if(!lastInstruction) return;
taskInput.value=lastInstruction;
await startTask();
}

async function pollStatus(){
if(!currentTaskId) return;
try{
const response=await fetch('/ai/status/'+currentTaskId);
const data=await response.json();
if(data){
updateTaskStatus(data);
if(data.ai_cursor_pos){
cursorInfo.classList.add('active');
const x=data.ai_cursor_pos[0];
const y=data.ai_cursor_pos[1];
cursorCoords.innerHTML='<span class="label">AI Cursor:</span><span class="value">X: '+x+', Y: '+y+'</span><div class="cursor-status">🎯 Moving...</div>';
}else{
cursorInfo.classList.remove('active');
}
if(data.status==='completed'){
msg.textContent='✅ Task completed!';
msg.style.background='rgba(16,185,129,0.95)';
setTimeout(function(){msg.style.display='none';msg.style.background='rgba(74,158,255,0.95)'},3000);
currentTaskId=null;
startBtn.style.display='block';
stopBtn.style.display='none';
retryBtn.style.display='block';
startBtn.disabled=false;
cursorInfo.classList.remove('active');
}else if(data.status==='failed' || data.status==='stopped'){
msg.textContent=(data.status==='failed'?'❌ Task failed: ':'⚠️ Task stopped: ')+(data.error||'');
msg.style.background='rgba(239,68,68,0.95)';
setTimeout(function(){msg.style.display='none';msg.style.background='rgba(74,158,255,0.95)'},5000);
currentTaskId=null;
startBtn.style.display='block';
stopBtn.style.display='none';
retryBtn.style.display='block';
startBtn.disabled=false;
cursorInfo.classList.remove('active');
}
}
}catch(e){
console.error(e);
}
}

function updateTaskStatus(data){
const status=document.getElementById('taskStatus');
const statusClass=data.status==='running'?'status-running':data.status==='completed'?'status-completed':data.status==='stopped'?'status-stopped':'status-failed';
const planCount=data.plan_steps?data.plan_steps.length:0;
const planProgress=data.plan_progress||0;
status.innerHTML='<div class="status-item '+statusClass+'"><div><strong>Status:</strong> '+data.status.toUpperCase()+'</div><div><strong>Steps:</strong> '+data.current_step+'/'+data.max_steps+'</div><div><strong>Plan:</strong> '+planProgress+'/'+planCount+'</div><div><strong>Instruction:</strong> '+data.instruction+'</div></div>';
if(data.logs && data.logs.length>0){
const logsDiv=document.getElementById('taskLogs');
logsDiv.innerHTML=data.logs.map(function(log){
const cls=log.includes('Clicked')||log.includes('Typed')||log.includes('completed')?'success':log.includes('failed')?'error':'info';
return '<div class="log-entry '+cls+'">'+log+'</div>';
}).join('');
logsDiv.scrollTop=logsDiv.scrollHeight;
}
renderPlan(data);
}

function addLog(type,message){
const logsDiv=document.getElementById('taskLogs');
const entry=document.createElement('div');
entry.className='log-entry '+type;
entry.textContent=message;
logsDiv.appendChild(entry);
logsDiv.scrollTop=logsDiv.scrollHeight;
}

function renderPlan(data){
const panel=document.getElementById('planPanel');
const list=document.getElementById('planList');
const label=document.getElementById('planProgressLabel');
if(!panel || !list || !label) return;
const steps=data.plan_steps||[];
const progress=data.plan_progress||0;
label.textContent=steps.length?progress+'/'+steps.length:'0/0';
if(!steps.length){
list.innerHTML='<li class="plan-step next"><span class="plan-step-index">-</span><span>No structured plan available.</span></li>';
return;
}
list.innerHTML=steps.map(function(step,index){
const stepNumber=index+1;
let cls='plan-step next';
if(stepNumber<=progress){
cls='plan-step done';
}else if(stepNumber===progress+1){
cls='plan-step current';
}
return '<li class="'+cls+'"><span class="plan-step-index">'+stepNumber+'.</span><span>'+step+'</span></li>';
}).join('');
}

setInterval(update,150);
setInterval(pollStatus,500);
checkOllama();
setInterval(checkOllama,5000);
setTimeout(update,300);
</script></body></html>'''
    return render_template_string(html)

@app.route('/stream')
def stream():
    global frame_size, last_stream_request
    last_stream_request = time.time()
    if current_frame:
        response = Response(current_frame, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['X-Frame-Width'] = str(frame_size[0])
        response.headers['X-Frame-Height'] = str(frame_size[1])
        return response
    img = Image.new('RGB', (1280,800), 'black')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img.close()
    response = Response(buf.getvalue(), mimetype='image/jpeg')
    response.headers['X-Frame-Width'] = str(frame_size[0])
    response.headers['X-Frame-Height'] = str(frame_size[1])
    return response

@app.route('/ai/health')
def ai_health():
    model_ready = vlm_model is not None
    backend_error = vlm_last_error
    status = {
        'ollama_available': backend_error is None,
        'model': VLM_MODEL_ID,
        'model_available': model_ready,
        'device': str(vlm_device)
    }
    if backend_error:
        status['error'] = backend_error
    return jsonify(status)


@app.route('/favicon.ico')
def favicon():
    # ブラウザの404を避けるためのプレースホルダー応答
    return ('', 204)

@app.route('/ai/task', methods=['POST'])
def create_ai_task():
    data = request.json
    instruction = data.get('instruction', '').strip()
    if not instruction:
        return jsonify({'error': 'Instruction required'}), 400
    task_id = f"task_{int(time.time() * 1000)}"
    task = AITask(task_id=task_id, instruction=instruction, status=TaskStatus.QUEUED)
    task.plan_steps = build_task_plan(instruction)
    if task.plan_steps:
        plan_preview = "; ".join(task.plan_steps[:4])
        task.logs.append(f"🧭 Plan initialized: {plan_preview}...")
    with ai_task_lock:
        ai_tasks[task_id] = task
    threading.Thread(target=ai_agent_loop, args=(task_id,), daemon=True).start()
    return jsonify({'task_id': task_id, 'status': 'queued'})

@app.route('/ai/status/<task_id>')
def get_task_status(task_id):
    with ai_task_lock:
        if task_id not in ai_tasks:
            return jsonify({'error': 'Task not found'}), 404
        task = ai_tasks[task_id]
        cursor_pos = ai_cursor_position or ai_last_cursor_position
        return jsonify({
            'task_id': task.task_id,
            'instruction': task.instruction,
            'status': task.status.value,
            'current_step': task.current_step,
            'max_steps': task.max_steps,
            'plan_steps': task.plan_steps,
            'plan_progress': task.plan_progress,
            'logs': task.logs[-20:],
            'error': task.error,
            'ai_cursor_pos': cursor_pos
        })

@app.route('/ai/stop/<task_id>', methods=['POST'])
def stop_ai_task(task_id):
    global ai_cursor_position
    with ai_task_lock:
        if task_id not in ai_tasks:
            return jsonify({'error': 'Task not found'}), 404
        task = ai_tasks[task_id]
        if task.status == TaskStatus.RUNNING:
            task.status = TaskStatus.STOPPED
            task.completed_at = time.time()
            ai_cursor_position = None
            return jsonify({'status': 'stopped'})
        return jsonify({'status': task.status.value})

@app.after_request
def add_no_cache_headers(response):
    response.headers.setdefault('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    response.headers.setdefault('Pragma', 'no-cache')
    response.headers.setdefault('Expires', '0')
    return response

if __name__=='__main__':
    print("="*60)
    print("AI Browser Agent")
    print("="*60)
    print("Starting browser...")
    init_browser()
    threading.Thread(target=capture_loop, daemon=True).start()
    print("Capture loop started")
    print("Loading local vision-language model (first load may download several GB)...")
    try:
        load_vlm()
        print(f"Vision-language model ready on device: {vlm_device}")
    except Exception as preload_error:
        print(f"⚠️  Failed to preload vision-language model: {preload_error}")
        print("The application will continue, but AI tasks will not run until the model loads successfully.")
    print("\nOpen http://127.0.0.1:5000")
    print("="*60)
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    finally:
        running = False
        if driver:
            driver.quit()
