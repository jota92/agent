from flask import Flask, Response, render_template_string, request, jsonify
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException
import threading
import time
import io
from PIL import Image

app = Flask(__name__)
driver = None
lock = threading.Lock()
current_frame = None
running = True
frame_size = (1280, 800)
last_stream_request = 0.0

JPEG_QUALITY = 60
TARGET_STREAM_WIDTH = 1024
ACTIVE_CAPTURE_DELAY = 0.1
IDLE_CAPTURE_DELAY = 0.65
IDLE_THRESHOLD_SECONDS = 2.0
RESAMPLE_FILTER = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.LANCZOS)

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
    print(f"Restarting browser due to: {reason}")
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

def capture_loop():
    global current_frame, running, frame_size, last_stream_request
    while running:
        try:
            now = time.time()
            if now - last_stream_request > IDLE_THRESHOLD_SECONDS:
                time.sleep(IDLE_CAPTURE_DELAY)
                continue
            png = None
            # grab the raw PNG under lock as fast as possible, then do processing outside
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
                # process image outside of the lock to avoid blocking handlers
                image = Image.open(io.BytesIO(png))
                frame_size = image.size
                processed = image
                if TARGET_STREAM_WIDTH and image.width > TARGET_STREAM_WIDTH:
                    ratio = TARGET_STREAM_WIDTH / float(image.width)
                    new_height = max(1, int(image.height * ratio))
                    processed = image.resize((TARGET_STREAM_WIDTH, new_height), RESAMPLE_FILTER)
                jpeg_buffer = io.BytesIO()
                processed.convert('RGB').save(jpeg_buffer, format='JPEG', quality=JPEG_QUALITY)
                # set current_frame quickly
                with lock:
                    current_frame = jpeg_buffer.getvalue()
                if processed is not image:
                    processed.close()
                image.close()
            time.sleep(ACTIVE_CAPTURE_DELAY)
        except Exception as e:
            print(f"Capture: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template_string('''
<html><head><title>Browser</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#000;height:100vh;display:flex;flex-direction:column;font-family:Arial}
.bar{background:#333;padding:12px;display:flex;gap:10px}
.tabs{background:#1f1f1f;padding:8px 12px;display:flex;gap:8px;overflow-x:auto}
.tab{padding:6px 12px;border-radius:12px;background:#2c2c2c;color:#ccc;border:none;cursor:pointer;flex:0 0 auto}
.tab.active{background:#4a8af4;color:#fff}
.tab.add{background:#3d3d3d;color:#90caf9;font-weight:bold}
input{flex:1;padding:10px;border-radius:20px;background:#444;color:#fff;border:none}
button{padding:10px 20px;border-radius:20px;background:#1a73e8;color:#fff;border:none;cursor:pointer}
.view{flex:1;display:flex;align-items:center;justify-content:center;overflow:auto;position:relative}
img{max-width:100%;cursor:pointer}
.msg{position:absolute;top:20px;background:rgba(0,0,0,0.8);color:#fff;padding:10px 20px;border-radius:8px;display:none}
#ime-buffer{position:fixed;left:4px;top:4px;width:1px;height:1px;opacity:0;border:0;padding:0;background:transparent;color:transparent;z-index:10000}
</style></head><body>
<div class="bar">
<input id="url" value="google.com">
<button id="go">Go</button>
<button id="back">Back</button>
</div>
<div class="tabs" id="tabs"></div>
<div class="view">
<div class="msg" id="msg">Loading...</div>
<img id="img" tabindex="0">
</div>
<textarea id="ime-buffer" autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false"></textarea>
<script>
const url=document.getElementById('url');
const img=document.getElementById('img');
const msg=document.getElementById('msg');
const tabs=document.getElementById('tabs');
const imeBuffer=document.getElementById('ime-buffer');
let remoteFocus=false;
let fetchingStream=false;
let lastObjectUrl=null;
let serverFrameWidth=1280;
let serverFrameHeight=800;
const ACTIVE_INTERVAL=120;
const PASSIVE_INTERVAL=420;
const HIDDEN_INTERVAL=1200;
const TAB_INTERVAL=3600;

img.onload=()=>{msg.style.display='none';};
img.onerror=err=>{console.log(err);};

async function update(){
if(fetchingStream) return;
fetchingStream=true;
try{
const response=await fetch('/stream?'+Date.now(),{cache:'no-store'});
if(!response.ok) throw new Error('stream '+response.status);
const serverW=response.headers.get('X-Frame-Width');
const serverH=response.headers.get('X-Frame-Height');
if(serverW) serverFrameWidth=parseInt(serverW,10)||serverFrameWidth;
if(serverH) serverFrameHeight=parseInt(serverH,10)||serverFrameHeight;
const blob=await response.blob();
const objectUrl=URL.createObjectURL(blob);
if(lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
lastObjectUrl=objectUrl;
img.src=objectUrl;
}catch(err){
console.log(err);
}finally{
fetchingStream=false;
}
}

function ensureImeFocus(){
if(!remoteFocus) return;
if(document.activeElement!==imeBuffer){
imeBuffer.focus();
}
}

function activateRemoteFocus(){
remoteFocus=true;
ensureImeFocus();
}

function deactivateRemoteFocus(){
remoteFocus=false;
imeBuffer.value='';
if(document.activeElement===imeBuffer){
imeBuffer.blur();
}
}

function sendTextInsert(text){
if(!text) return;
fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({type:'insert',text})});
update();
}

// ensure an actual caret is placed so IME composition starts reliably
imeBuffer.addEventListener('focus',()=>{
    try{
        imeBuffer.setSelectionRange(imeBuffer.value.length, imeBuffer.value.length);
    }catch(e){}
});

async function sleep(ms){return new Promise(res=>setTimeout(res,ms));}

async function updateLoop(){
while(true){
await update();
if(remoteFocus) ensureImeFocus();
const delay=document.hidden?HIDDEN_INTERVAL:(remoteFocus?ACTIVE_INTERVAL:PASSIVE_INTERVAL);
await sleep(delay);
}
}

function nav(u){
if(!u.startsWith('http'))u='https://'+u;
msg.textContent='Loading...';msg.style.display='block';
fetch('/nav',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:u})})
.then(r=>r.json())
.then(d=>{
url.value=d.url;
if(!d.success&&d.error){console.log(d.error);}
update();
setTimeout(loadTabs,320);
});
}

url.addEventListener('keypress',e=>{if(e.key==='Enter')nav(url.value)});
document.getElementById('go').onclick=()=>nav(url.value);
document.getElementById('back').onclick=()=>fetch('/back',{method:'POST'})
.then(()=>{update();setTimeout(loadTabs,260);});

img.onclick=e=>{
const r=img.getBoundingClientRect();
const rectWidth=Math.max(1,r.width);
const rectHeight=Math.max(1,r.height);
const relX=(e.clientX-r.left)/rectWidth;
const relY=(e.clientY-r.top)/rectHeight;
const x=Math.max(0,Math.round(relX*serverFrameWidth));
const y=Math.max(0,Math.round(relY*serverFrameHeight));
fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({x,y})});
img.focus();
activateRemoteFocus();
update();
};

img.addEventListener('focus',()=>{activateRemoteFocus();});
img.addEventListener('blur',()=>{
if(document.activeElement===imeBuffer) return;
deactivateRemoteFocus();
});
imeBuffer.addEventListener('blur',()=>{
if(document.activeElement===img) return;
deactivateRemoteFocus();
});
document.addEventListener('visibilitychange',()=>{if(document.hidden)deactivateRemoteFocus();});
url.addEventListener('focus',()=>{deactivateRemoteFocus();});

function sendKey(event,type){
const payload={
type,
key:event.key,
code:event.code,
text:event.key.length===1?event.key:'',
keyCode:event.keyCode||event.which||0,
repeat:event.repeat||false,
shift:event.shiftKey||false,
ctrl:event.ctrlKey||false,
alt:event.altKey||false,
meta:event.metaKey||false
};
fetch('/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
}

imeBuffer.addEventListener('keydown',e=>{
if(!remoteFocus) return;
if(e.key==='Meta') return;
const printable=e.key.length===1 && !e.ctrlKey && !e.metaKey && !e.altKey;
if(!printable) e.preventDefault();
sendKey(e,'keyDown');
});

imeBuffer.addEventListener('keyup',e=>{
if(!remoteFocus) return;
if(e.key==='Meta') return;
e.preventDefault();
sendKey(e,'keyUp');
});

imeBuffer.addEventListener('input',()=>{
if(!remoteFocus) return;
const text=imeBuffer.value;
if(!text) return;
sendTextInsert(text);
imeBuffer.value='';
});

imeBuffer.addEventListener('compositionend',()=>{
if(!remoteFocus) return;
const text=imeBuffer.value;
if(!text) return;
sendTextInsert(text);
imeBuffer.value='';
});

imeBuffer.addEventListener('compositionstart',()=>{
    // ensure buffer is empty at composition start
    imeBuffer.value='';
});

imeBuffer.addEventListener('paste',e=>{
if(!remoteFocus) return;
e.preventDefault();
const text=(e.clipboardData||window.clipboardData).getData('text');
sendTextInsert(text);
});

document.addEventListener('pointerdown',e=>{
if(remoteFocus && !e.composedPath().includes(img) && e.target!==imeBuffer){
deactivateRemoteFocus();
}
});

function loadTabs(){
if(document.hidden) return;
fetch('/tabs?'+Date.now())
.then(r=>r.json())
.then(d=>{
tabs.innerHTML='';
const addButton=document.createElement('button');
addButton.className='tab add';
addButton.textContent='+';
addButton.title='新しいタブを開く';
addButton.onclick=()=>createTab();
tabs.appendChild(addButton);
d.tabs.forEach(tab=>{
const btn=document.createElement('button');
btn.className='tab'+(tab.active?' active':'');
btn.textContent=tab.title||'(No title)';
btn.title=tab.url;
btn.onclick=()=>switchTab(tab.handle);
tabs.appendChild(btn);
});
})
.catch(e=>console.log(e));
}

function switchTab(handle){
fetch('/switch_tab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({handle})})
.then(r=>r.json())
.then(d=>{if(d.success){url.value=d.url;msg.textContent='Loading...';msg.style.display='block';setTimeout(()=>{update();loadTabs();},300);}})
.catch(e=>console.log(e));
}

function createTab(){
msg.textContent='新しいタブを開いています...';
msg.style.display='block';
fetch('/new_tab',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:url.value})})
.then(r=>r.json())
.then(d=>{
if(d.success){
url.value=d.url;
setTimeout(()=>{update();loadTabs();},400);
}else if(d.error){
console.log(d.error);
msg.style.display='none';
}
})
.catch(e=>{console.log(e);msg.style.display='none';});
}

updateLoop();
setInterval(loadTabs,TAB_INTERVAL);
setTimeout(()=>{
fetch('/url').then(r=>r.json()).then(d=>{url.value=d.url;update();loadTabs()});
},500);
</script></body></html>
''')

@app.route('/stream')
def stream():
    global frame_size, last_stream_request
    last_stream_request = time.time()
    if current_frame:
        response = Response(current_frame, mimetype='image/jpeg')
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Frame-Width'] = str(frame_size[0])
        response.headers['X-Frame-Height'] = str(frame_size[1])
        return response
    img = Image.new('RGB', (1280,800), 'black')
    frame_size = img.size
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    img.close()
    response = Response(buf.getvalue(), mimetype='image/jpeg')
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['X-Frame-Width'] = str(frame_size[0])
    response.headers['X-Frame-Height'] = str(frame_size[1])
    return response

@app.route('/nav', methods=['POST'])
def navigate():
    global last_stream_request
    data = request.json
    with lock:
        try:
            if not driver:
                init_browser()
            driver.get(data['url'])
            time.sleep(0.5)
            last_stream_request = time.time()
            return jsonify({'success':True,'url':driver.current_url})
        except Exception as e:
            restart_browser_locked(f"navigation error: {e}")
            return jsonify({'success':False,'error':str(e)})

@app.route('/new_tab', methods=['POST'])
def new_tab():
    global last_stream_request
    data = request.json or {}
    target_url = data.get('url', 'https://www.google.com')
    with lock:
        try:
            if not driver:
                init_browser()
            before_handles = set(driver.window_handles)
            driver.execute_script("window.open(arguments[0], '_blank');", target_url)
            time.sleep(0.35)
            handles = driver.window_handles
            new_handles = [h for h in handles if h not in before_handles]
            if new_handles:
                driver.switch_to.window(new_handles[-1])
            else:
                driver.switch_to.window(handles[-1])
            current_url = driver.current_url
            last_stream_request = time.time()
            return jsonify({'success': True, 'url': current_url})
        except Exception as e:
            restart_browser_locked(f"new tab error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/click', methods=['POST'])
def click():
    global last_stream_request
    data = request.json or {}
    x = data.get('x')
    y = data.get('y')
    if x is None or y is None:
        return jsonify({'success':False,'error':'Invalid coordinates'}), 400
    with lock:
        try:
            if not driver:
                init_browser()
            metrics = driver.execute_cdp_cmd('Page.getLayoutMetrics', {})
            layout = metrics.get('layoutViewport', {})
            visual = metrics.get('visualViewport', {})
            layout_width = visual.get('clientWidth') or layout.get('clientWidth', frame_size[0])
            layout_height = visual.get('clientHeight') or layout.get('clientHeight', frame_size[1])
            scale_x = layout_width / frame_size[0] if frame_size[0] else 1
            scale_y = layout_height / frame_size[1] if frame_size[1] else 1
            target_x = x * scale_x
            target_y = y * scale_y
            driver.execute_cdp_cmd('Input.dispatchMouseEvent', {
                'type': 'mouseMoved',
                'x': target_x,
                'y': target_y,
                'buttons': 1
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
            last_stream_request = time.time()
            return jsonify({'success':True})
        except Exception as e:
            print(f"Click: {e}")
            restart_browser_locked(f"click error: {e}")
            return jsonify({'success':False,'error':str(e)})

@app.route('/back', methods=['POST'])
def back():
    global last_stream_request
    with lock:
        try:
            if not driver:
                init_browser()
            driver.back()
            last_stream_request = time.time()
            return jsonify({'success':True})
        except Exception as e:
            restart_browser_locked(f"back error: {e}")
            return jsonify({'success':False})

@app.route('/url')
def get_url():
    with lock:
        try:
            if not driver:
                init_browser()
            return jsonify({'url':driver.current_url if driver else 'about:blank'})
        except Exception:
            return jsonify({'url':'about:blank'})

@app.route('/tabs')
def tabs():
    with lock:
        try:
            if not driver:
                init_browser()
            if not driver:
                return jsonify({'tabs': []})
            current = driver.current_window_handle
            handles = driver.window_handles
            tabs_payload = []
            for handle in handles:
                driver.switch_to.window(handle)
                tabs_payload.append({
                    'handle': handle,
                    'title': driver.title,
                    'url': driver.current_url,
                    'active': handle == current
                })
            driver.switch_to.window(current)
            return jsonify({'tabs': tabs_payload})
        except Exception as e:
            restart_browser_locked(f"tabs error: {e}")
            return jsonify({'tabs': [], 'error': str(e)}), 500

@app.route('/switch_tab', methods=['POST'])
def switch_tab():
    global last_stream_request
    data = request.json or {}
    handle = data.get('handle')
    if not handle:
        return jsonify({'success':False,'error':'Missing handle'}), 400
    with lock:
        try:
            if not driver:
                init_browser()
            if not driver:
                return jsonify({'success':False,'error':'Browser not available'}), 400
            handles = driver.window_handles
            if handle not in handles:
                return jsonify({'success':False,'error':'Tab not found'}), 404
            driver.switch_to.window(handle)
            current_url = driver.current_url
            last_stream_request = time.time()
            return jsonify({'success':True,'url':current_url})
        except Exception as e:
            restart_browser_locked(f"switch tab error: {e}")
            return jsonify({'success':False,'error':str(e)}), 500

def _compute_modifiers(data):
    mods = 0
    if data.get('shift'):
        mods |= 1
    if data.get('ctrl'):
        mods |= 2
    if data.get('alt'):
        mods |= 4
    if data.get('meta'):
        mods |= 8
    return mods

SPECIAL_KEYS = {'Enter': 13, 'Backspace': 8, 'Tab': 9, 'Escape': 27, 'ArrowLeft': 37,
                'ArrowUp': 38, 'ArrowRight': 39, 'ArrowDown': 40, 'Delete': 46}

@app.route('/key', methods=['POST'])
def key_event():
    global last_stream_request
    data = request.json or {}
    key = data.get('key')

    event_type = data.get('type')
    if not key or not event_type:
        return jsonify({'success': False, 'error': 'Invalid key event'}), 400
    with lock:
        try:
            if not driver:
                init_browser()
            if not driver:
                return jsonify({'success': False, 'error': 'Browser not available'}), 400
            modifiers = _compute_modifiers(data)
            key_code = data.get('keyCode') or SPECIAL_KEYS.get(key) or (ord(key) if len(key) == 1 else 0)
            if event_type == 'insert' and len(data.get('text', '')) > 0:
                driver.execute_cdp_cmd('Input.insertText', {'text': data['text']})
            else:
                driver.execute_cdp_cmd('Input.dispatchKeyEvent', {
                    'type': event_type,
                    'key': key,
                    'code': data.get('code') or key,
                    'text': data.get('text', ''),
                    'unmodifiedText': data.get('text', ''),
                    'windowsVirtualKeyCode': key_code,
                    'nativeVirtualKeyCode': key_code,
                    'modifiers': modifiers,
                    'repeat': data.get('repeat', False)
                })
            last_stream_request = time.time()
            return jsonify({'success': True})
        except Exception as e:
            print(f"Key: {e}")
            restart_browser_locked(f"key error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

@app.after_request
def add_no_cache_headers(response):
    response.headers.setdefault('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
    response.headers.setdefault('Pragma', 'no-cache')
    response.headers.setdefault('Expires', '0')
    return response

if __name__=='__main__':
    print("Starting browser...")
    init_browser()
    threading.Thread(target=capture_loop, daemon=True).start()
    print("Open http://127.0.0.1:5000")
    try:
        app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    finally:
        running = False
        if driver:
            driver.quit()
