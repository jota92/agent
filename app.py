from flask import Flask, Response, render_template_string, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import threading
import time
import io
import json
import base64
import requests
from PIL import Image, ImageDraw
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
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2-vision:11b"
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

    def __post_init__(self):
        if self.logs is None:
            self.logs = []
        if self.history is None:
            self.history = []
        if self.created_at == 0:
            self.created_at = time.time()

ai_tasks: Dict[str, AITask] = {}
ai_task_lock = threading.Lock()
ai_cursor_position = None

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
    service = Service('/opt/homebrew/bin/chromedriver')
    try:
        driver_instance = webdriver.Chrome(service=service, options=options)
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
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    width, height = img.size
    
    # グリッド線の間隔
    grid_spacing = 100
    
    # 縦線
    for x in range(0, width, grid_spacing):
        # 薄いグレーの線
        draw.line([(x, 0), (x, height)], fill=(100, 100, 100, 80), width=1)
        # 座標ラベル（上端）
        if x > 0:
            draw.text((x + 2, 5), str(x), fill=(255, 255, 0, 200))
    
    # 横線
    for y in range(0, height, grid_spacing):
        # 薄いグレーの線
        draw.line([(0, y), (width, y)], fill=(100, 100, 100, 80), width=1)
        # 座標ラベル（左端）
        if y > 0:
            draw.text((5, y + 2), str(y), fill=(255, 255, 0, 200))
    
    # 中心マーカー（特に目立つ）
    center_x, center_y = width // 2, height // 2
    draw.line([(center_x - 20, center_y), (center_x + 20, center_y)], fill=(0, 255, 0, 150), width=3)
    draw.line([(center_x, center_y - 20), (center_x, center_y + 20)], fill=(0, 255, 0, 150), width=3)
    draw.text((center_x + 25, center_y - 10), f"({center_x}, {center_y})", fill=(0, 255, 0, 200))
    
    return img_copy

def draw_ai_cursor(img: Image.Image, pos: tuple) -> Image.Image:
    """極端に大きく目立つAIカーソルを描画"""
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy, 'RGBA')
    x, y = pos
    
    # 超巨大外側グロー（パルス効果）
    glow_radius = 80
    glow_color = (255, 120, 0, 80)
    draw.ellipse([(x - glow_radius, y - glow_radius), (x + glow_radius, y + glow_radius)], 
                 fill=glow_color)
    
    # 大きな中間グロー
    mid_glow = 60
    mid_color = (255, 150, 50, 120)
    draw.ellipse([(x - mid_glow, y - mid_glow), (x + mid_glow, y + mid_glow)], 
                 fill=mid_color)
    
    # 内側グロー
    inner_glow = 40
    inner_color = (255, 200, 100, 160)
    draw.ellipse([(x - inner_glow, y - inner_glow), (x + inner_glow, y + inner_glow)], 
                 fill=inner_color)
    
    # カーソル本体の影（立体感）
    shadow_offset = 5
    shadow_color = (0, 0, 0, 200)
    size = 60  # 極端に大きく
    thickness = 10  # 極端に太く
    
    # 影のクロスヘア
    draw.line([(x + shadow_offset, y - size + shadow_offset), (x + shadow_offset, y + size + shadow_offset)], 
             fill=shadow_color, width=thickness)
    draw.line([(x - size + shadow_offset, y + shadow_offset), (x + size + shadow_offset, y + shadow_offset)], 
             fill=shadow_color, width=thickness)
    
    # メインのクロスヘア（極端に太く鮮やか）
    main_color = (255, 0, 0, 255)  # 真っ赤
    outline_color = (255, 255, 255, 255)
    
    # 極太白い縁取り
    draw.line([(x, y - size), (x, y + size)], fill=outline_color, width=thickness + 4)
    draw.line([(x - size, y), (x + size, y)], fill=outline_color, width=thickness + 4)
    
    # メインカラー
    draw.line([(x, y - size), (x, y + size)], fill=main_color, width=thickness)
    draw.line([(x - size, y), (x + size, y)], fill=main_color, width=thickness)
    
    # 外側の超大きな円
    circle_radius = 35
    draw.ellipse([(x - circle_radius, y - circle_radius), (x + circle_radius, y + circle_radius)], 
                outline=outline_color, width=thickness)
    draw.ellipse([(x - circle_radius - 2, y - circle_radius - 2), (x + circle_radius + 2, y + circle_radius + 2)], 
                outline=main_color, width=thickness - 2)
    
    # 中心の超大きな点（ターゲット）
    center_dot = 15
    draw.ellipse([(x - center_dot, y - center_dot), (x + center_dot, y + center_dot)], 
                fill=(255, 255, 0, 255), outline=(255, 150, 0, 255), width=4)
    
    return img_copy

def capture_loop():
    global current_frame, running, frame_size, last_stream_request, ai_cursor_position
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
                    cursor_pos = ai_cursor_position
                
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

def call_ollama_vision(image_b64: str, instruction: str, step_num: int, history: List[Dict] = None) -> Optional[AIAction]:
    if history is None:
        history = []
    
    # 直近3ステップの履歴を整形
    history_text = ""
    last_action_type = None
    if history:
        history_text = "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n**YOUR RECENT ACTIONS (LAST 3 STEPS):**\n"
        for h in history[-3:]:
            act_dict = h.get('action', {})
            action_name = act_dict.get('action', 'unknown')
            reasoning = act_dict.get('reasoning', 'N/A')
            x = act_dict.get('x', '')
            y = act_dict.get('y', '')
            coords = f" at ({x},{y})" if x and y else ""
            history_text += f"✓ Step {h['step']}: **{action_name.upper()}**{coords} - {reasoning}\n"
        
        # 最後のアクションタイプを取得
        if history:
            last_action_type = history[-1].get('action', {}).get('action')
        
        history_text += "\n⚠️ **ABSOLUTE RULE**: "
        if last_action_type == "click":
            history_text += "You JUST CLICKED! Your next action MUST be 'type' (NOT click again!)\n"
        elif last_action_type == "type":
            history_text += "You JUST TYPED! Your next action should be 'wait' or check results\n"
        elif last_action_type == "wait":
            history_text += "You JUST WAITED! Check if page loaded, then click result or done\n"
        history_text += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    else:
        history_text = "\n**This is your FIRST action. Start by clicking the search box.**\n"
    
    
    prompt = f"""You are controlling a web browser. Task: {instruction}

Current step: {step_num}
{history_text}

CRITICAL RULES:
1. The screen has a YELLOW GRID with coordinates labeled
2. Read the coordinates from the grid to find exact positions
3. Search box is typically at y=300-400, center x=640
4. You MUST respond with ONLY valid JSON - no other text before or after

REQUIRED SEQUENCE:
Step 1: Click search box → Step 2: Type query → Step 3: Wait → Step 4: Click result/done

Valid JSON formats (choose ONE, return ONLY the JSON):
{{"action": "click", "x": 640, "y": 350, "reasoning": "clicking search box at grid coordinates"}}
{{"action": "type", "text": "your query\\n", "reasoning": "typing search with enter"}}
{{"action": "wait", "reasoning": "waiting for page load"}}
{{"action": "done", "reasoning": "task completed"}}

Look at the screenshot with YELLOW GRID. What is the next action? Return ONLY JSON:"""

    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "images": [image_b64], "stream": False, "options": {"temperature": 0.05, "top_p": 0.7, "top_k": 10}}
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        response_text = result.get('response', '').strip()
        
        print(f"🤖 AI Response: {response_text[:300]}")  # デバッグログ
        
        # より柔軟なJSON抽出
        if '{' in response_text and '}' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            try:
                action_data = json.loads(json_str)
                print(f"✓ Parsed JSON: {action_data}")
                return AIAction(**action_data)
            except json.JSONDecodeError as je:
                print(f"JSON parse error: {je}")
                print(f"Attempted to parse: {json_str}")
                return None
        else:
            print(f"❌ No JSON braces found in response: {response_text[:300]}")
            return None
    except Exception as e:
        print(f"❌ Ollama API error: {e}")
        import traceback
        traceback.print_exc()
        return None

def execute_ai_action(action: AIAction, task: AITask) -> bool:
    global ai_cursor_position, last_stream_request
    try:
        with lock:
            if not driver:
                init_browser()
            last_stream_request = time.time()
            if action.action == "click" and action.x is not None and action.y is not None:
                # カーソルを現在位置から目標位置まで滑らかにアニメーション
                start_pos = ai_cursor_position if ai_cursor_position else (frame_size[0] // 2, frame_size[1] // 2)
                end_pos = (action.x, action.y)
                
                # アニメーションステップ数（滑らかさ）
                animation_steps = 20
                animation_delay = 0.03  # 各ステップ間の遅延（秒）
                
                # カーソルを移動
                for step in range(animation_steps + 1):
                    progress = step / animation_steps
                    # イージング関数（加速→減速）
                    eased_progress = progress * progress * (3 - 2 * progress)  # smoothstep
                    
                    current_x = int(start_pos[0] + (end_pos[0] - start_pos[0]) * eased_progress)
                    current_y = int(start_pos[1] + (end_pos[1] - start_pos[1]) * eased_progress)
                    
                    with ai_task_lock:
                        ai_cursor_position = (current_x, current_y)
                    
                    time.sleep(animation_delay)
                
                # 最終位置で少し停止（クリック準備）
                time.sleep(0.3)
                
                # クリック実行
                metrics = driver.execute_cdp_cmd('Page.getLayoutMetrics', {})
                layout = metrics.get('layoutViewport', {})
                visual = metrics.get('visualViewport', {})
                layout_width = visual.get('clientWidth') or layout.get('clientWidth', frame_size[0])
                layout_height = visual.get('clientHeight') or layout.get('clientHeight', frame_size[1])
                scale_x = layout_width / frame_size[0] if frame_size[0] else 1
                scale_y = layout_height / frame_size[1] if frame_size[1] else 1
                target_x = action.x * scale_x
                target_y = action.y * scale_y
                
                # マウスダウン→アップ（クリックエフェクト）
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {'type': 'mousePressed', 'x': target_x, 'y': target_y, 'button': 'left', 'clickCount': 1})
                time.sleep(0.1)  # クリックの視覚的フィードバック
                driver.execute_cdp_cmd('Input.dispatchMouseEvent', {'type': 'mouseReleased', 'x': target_x, 'y': target_y, 'button': 'left', 'clickCount': 1})
                
                task.logs.append(f"✓ Clicked at ({action.x}, {action.y}): {action.reasoning}")
                
                # クリック後にカーソルを少し表示してからクリア
                time.sleep(0.5)
                with ai_task_lock:
                    ai_cursor_position = None
                    
            elif action.action == "type" and action.text:
                # 入力位置を示すためにカーソルを表示（中央付近）
                if not ai_cursor_position:
                    with ai_task_lock:
                        ai_cursor_position = (frame_size[0] // 2, frame_size[1] // 2)
                    time.sleep(0.3)
                
                # テキスト入力（Enterキーサポート）
                # \n を実際の改行文字として検出
                has_enter = '\n' in action.text or '\\n' in action.text
                
                if has_enter:
                    # Enterキーを含む場合 - \n または \\n を削除
                    text_without_enter = action.text.replace('\n', '').replace('\\n', '')
                    
                    # テキストがあれば入力
                    if text_without_enter:
                        driver.execute_cdp_cmd('Input.insertText', {'text': text_without_enter})
                        time.sleep(0.2)  # 入力が反映されるまで待機
                    
                    # Enterキーを送信
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
                    task.logs.append(f"✓ Typed: '{text_without_enter}' + ENTER - {action.reasoning}")
                    print(f"🔑 Enter key pressed after typing: {text_without_enter}")
                else:
                    driver.execute_cdp_cmd('Input.insertText', {'text': action.text})
                    task.logs.append(f"✓ Typed: '{action.text}' - {action.reasoning}")
                
                # 入力後カーソルをクリア
                time.sleep(0.5)
                with ai_task_lock:
                    ai_cursor_position = None
            elif action.action == "navigate" and action.url:
                driver.get(action.url)
                task.logs.append(f"✓ Navigated to: {action.url}")
            elif action.action == "scroll":
                driver.execute_script("window.scrollBy(0, 300)")
                task.logs.append(f"✓ Scrolled down - {action.reasoning}")
            elif action.action == "wait":
                time.sleep(2)
                task.logs.append(f"⏳ Waiting - {action.reasoning}")
            elif action.action == "done":
                task.logs.append(f"✅ Task completed - {action.reasoning}")
                with ai_task_lock:
                    ai_cursor_position = None
                return False
            time.sleep(0.5)
            return True
    except Exception as e:
        task.logs.append(f"Action failed: {e}")
        print(f"Execute action error: {e}")
        return True

def ai_agent_loop(task_id: str):
    global ai_cursor_position
    with ai_task_lock:
        if task_id not in ai_tasks:
            return
        task = ai_tasks[task_id]
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
    print(f"AI Agent starting task: {task.instruction}")
    
    last_action_signature = None  # 前回のアクション記録
    consecutive_click_count = 0  # 連続クリック回数
    last_click_pos = None  # 前回のクリック座標
    
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

            # 思考中カーソルを表示
            temp_image = Image.open(io.BytesIO(png))
            thinking_x, thinking_y = temp_image.width // 2, temp_image.height // 2
            temp_image.close()
            with ai_task_lock:
                ai_cursor_position = (thinking_x, thinking_y)
            
            image_b64 = base64.b64encode(png).decode('utf-8')
            print(f"Step {step + 1}: Asking AI for next action...")
            action = call_ollama_vision(image_b64, task.instruction, step + 1, task.history)
            
            # 思考中カーソルをクリア
            with ai_task_lock:
                ai_cursor_position = None
            time.sleep(0.1) # UI更新のための短い待機

            if not action:
                task.logs.append("AI failed to provide valid action")
                time.sleep(2)
                continue
            
            # より厳格な重複防止ロジック
            # 1. 完全一致の重複チェック
            action_signature = f"{action.action}:{action.x}:{action.y}:{action.text}"
            if action_signature == last_action_signature:
                task.logs.append(f"⚠️ Exact duplicate detected, skipping: {action.action}")
                print(f"⚠️ Prevented exact duplicate: {action_signature}")
                time.sleep(1)
                continue
            
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
                        import re
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
            if not should_continue:
                with ai_task_lock:
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    ai_cursor_position = None
                print(f"Task {task_id} completed successfully")
                break
            time.sleep(1)
        else:
            with ai_task_lock:
                task.status = TaskStatus.FAILED
                task.error = "Maximum steps reached"
                task.completed_at = time.time()
                ai_cursor_position = None
            print(f"Task {task_id} failed: max steps reached")
    except Exception as e:
        with ai_task_lock:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            ai_cursor_position = None
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
<div class="task-logs" id="taskLogs">
<div class="log-entry info">✓ Ready. Waiting for task...</div>
</div>
</div>
<div class="main">
<div class="browser-view">
<div class="msg" id="msg">Loading...</div>
<div class="ollama-status">
<div class="status-dot" id="ollamaDot"></div>
<span id="ollamaText">Checking Ollama...</span>
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
ollamaText.textContent='Ollama: '+data.model;
}else{
ollamaDot.className='status-dot disconnected';
ollamaText.textContent='Ollama: Disconnected';
}
}catch(e){
ollamaDot.className='status-dot disconnected';
ollamaText.textContent='Ollama: Error';
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
status.innerHTML='<div class="status-item '+statusClass+'"><div><strong>Status:</strong> '+data.status.toUpperCase()+'</div><div><strong>Step:</strong> '+data.current_step+'/'+data.max_steps+'</div><div><strong>Instruction:</strong> '+data.instruction+'</div></div>';
if(data.logs && data.logs.length>0){
const logsDiv=document.getElementById('taskLogs');
logsDiv.innerHTML=data.logs.map(function(log){
const cls=log.includes('Clicked')||log.includes('Typed')||log.includes('completed')?'success':log.includes('failed')?'error':'info';
return '<div class="log-entry '+cls+'">'+log+'</div>';
}).join('');
logsDiv.scrollTop=logsDiv.scrollHeight;
}
}

function addLog(type,message){
const logsDiv=document.getElementById('taskLogs');
const entry=document.createElement('div');
entry.className='log-entry '+type;
entry.textContent=message;
logsDiv.appendChild(entry);
logsDiv.scrollTop=logsDiv.scrollHeight;
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
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get('models', [])
            has_vision = any(OLLAMA_MODEL in str(m.get('name', '')) for m in models)
            return jsonify({'ollama_available': True, 'model': OLLAMA_MODEL, 'model_available': has_vision})
    except:
        pass
    return jsonify({'ollama_available': False, 'model': OLLAMA_MODEL, 'model_available': False})

@app.route('/ai/task', methods=['POST'])
def create_ai_task():
    data = request.json
    instruction = data.get('instruction', '').strip()
    if not instruction:
        return jsonify({'error': 'Instruction required'}), 400
    task_id = f"task_{int(time.time() * 1000)}"
    task = AITask(task_id=task_id, instruction=instruction, status=TaskStatus.QUEUED)
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
        cursor_pos = ai_cursor_position
        return jsonify({
            'task_id': task.task_id,
            'instruction': task.instruction,
            'status': task.status.value,
            'current_step': task.current_step,
            'max_steps': task.max_steps,
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
    print("\nOpen http://127.0.0.1:5000")
    print("\nMake sure Ollama is running:")
    print("  brew install ollama")
    print("  ollama serve")
    print("  ollama pull llava:13b")
    print("="*60)
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    finally:
        running = False
        if driver:
            driver.quit()
