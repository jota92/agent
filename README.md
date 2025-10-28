# Browser Automation Agent

Autonomous web browser control system powered by vision-language models. The agent analyzes screenshots, plans actions, and executes mouse clicks and keyboard inputs to complete web tasks.

## Features

- **Visual Understanding**: Uses Qwen2-VL-2B-Instruct to interpret web pages from screenshots
- **Task Planning**: Breaks down complex instructions into sequential steps
- **Real-time Monitoring**: Tracks DOM state and detects page changes
- **Coordinate Grid Overlay**: 100px grid with labels for precise navigation
- **Cursor Animation**: Smooth 20-step interpolation for natural movement
- **Loop Prevention**: Detects repeated actions and forces progress
- **Error Handling**: Auto-dismisses JavaScript dialogs and handles timeouts
- **Live Streaming**: Real-time browser view with plan progress tracking

## Requirements

- Linux or macOS
- Python 3.8+
- ChromeDriver

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/agent.git
cd agent
```

2. **Run setup script**
```bash
chmod +x start.sh
./start.sh
```

The script will:
- Create a Python virtual environment
- Install dependencies
- Download the Qwen2-VL-2B-Instruct model (first run only)
- Start the server

3. **Access the interface**
```
http://127.0.0.1:5000
```

## Usage

1. Open http://127.0.0.1:5000 in your browser
2. Enter a task in the text area (e.g., "Search for 'OpenAI' on Google and click the first result")
3. Click "Start Task"
4. Monitor progress in the sidebar and watch the cursor perform actions

## Technical Stack

- **Backend**: Flask (Python)
- **Browser Control**: Selenium WebDriver + Chrome DevTools Protocol
- **Vision Model**: Qwen2-VL-2B-Instruct (Transformers, PyTorch)
- **Image Processing**: Pillow
- **Frontend**: HTML/JavaScript with MJPEG streaming

## Architecture

### Cursor Visualization
- Animated pointer with glow effect
- Smooth interpolation between positions
- Overlay on coordinate grid

### Coordinate Grid
- Yellow 100px grid lines
- Axis labels and quadrant markers
- Center crosshair for reference

### Task Planner
- Extracts search queries from natural language
- Generates step-by-step execution plans
- Tracks completion progress

### Reliability Features
- Duplicate action detection
- Proximity-based click filtering (100px threshold)
- Forced action injection to break loops
- DOM readiness polling
- Page reload on visual stall detection
- Automatic alert dismissal

## API Endpoints

- `GET /` - Main web interface
- `GET /stream` - MJPEG video stream
- `POST /ai/task` - Start new automation task
- `GET /ai/status/<task_id>` - Get task status
- `POST /ai/stop/<task_id>` - Stop running task
- `GET /ai/health` - Check model availability

## Configuration

Customize behavior in `app.py`:

```python
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # Vision-language model
AI_MAX_STEPS = 30                            # Maximum steps per task
AI_STEP_TIMEOUT = 15                         # Step timeout in seconds
JPEG_QUALITY = 65                            # Stream image quality
TARGET_STREAM_WIDTH = 1024                   # Stream width in pixels
```

## License

MIT License

## Limitations

- Development server only - not suitable for production
- Autonomous actions may not always succeed
- Requires local execution for security

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss proposed modifications.
