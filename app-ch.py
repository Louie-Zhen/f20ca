from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
import base64
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import shutil
import time
from elevenlabs.client import ElevenLabs
from io import BytesIO

# Import utilities
from utils.llm import get_llm_response, build_booking_system_prompt
from utils.audio import convert_webm_to_wav
from utils.recording import save_recording_metadata
from utils.booking_state import get_or_create_session
from utils.calendar import initialize_calendar
from utils.vad import initialize_vad, validate_speech, trim_silence

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='.', template_folder='.')
app.config['SECRET_KEY'] = 'garage-booking-secret'

# Initialize SocketIO
# ping_timeout 设置长一点，防止生成语音时断开
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# ElevenLabs Client
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY not found!")
    raise ValueError("Please set ELEVENLABS_API_KEY in .env file")

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# LLM Provider Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'cohere').lower()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

logger.info(f"LLM Provider: {LLM_PROVIDER.upper()}")
logger.info("STT/TTS Provider: ELEVENLABS (Direct Backend)")

# Recording directories
RECORDINGS_DIR = 'recordings'
COMBINED_AUDIO_DIR = os.path.join(RECORDINGS_DIR, 'combined_audio')
METADATA_DIR = os.path.join(RECORDINGS_DIR, 'metadata')

for directory in [COMBINED_AUDIO_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

initialize_calendar()

# 尝试初始化 VAD，如果 torch 报错则忽略，避免卡死启动
try:
    initialize_vad()
except Exception as e:
    logger.warning(f"VAD initialization warning (likely torchcodec): {e}")
    logger.warning("Continuing without VAD optimization or using fallback...")

latency_records = []


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)


@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('status', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        start_time = time.time()
        # 初始化延迟记录器 (将不需要的步骤设为 0)
        latency_info = {
            'audio_conversion': 0.0,
            'vad_validation': 0.0,
            'silence_trimming': 0.0,
            'asr_transcription': 0,
            'llm_response': 0
        }

        audio_base64 = data.get('audio')
        if not audio_base64: return

        # 1. 仅解码 Base64，不进行任何格式转换！(保持 WebM 原始字节)
        audio_bytes = base64.b64decode(audio_base64)

        # 2. 直接发送给 ElevenLabs (极速模式)
        asr_start = time.time()
        logger.info("Transcribing WebM audio directly...")

        # 使用 BytesIO 包装字节流，并必须指定 .webm 后缀让 API 识别格式
        audio_buffer = BytesIO(audio_bytes)
        audio_buffer.name = "audio.webm"

        transcription_result = elevenlabs_client.speech_to_text.convert(
            file=audio_buffer,
            model_id="scribe_v2",
            language_code="cmn"  # 强制中文，避免语言检测耗时
        )
        transcription = transcription_result.text
        latency_info['asr_transcription'] = (time.time() - asr_start) * 1000
        logger.info(f"User said: {transcription}")

        # 如果 ElevenLabs 没听到声音，直接返回
        if not transcription or transcription.strip() == "":
            emit('bot_response', {'user_text': "...", 'bot_text': "（听不清）"})
            return

        # 3. LLM 生成
        llm_start = time.time()
        session = get_or_create_session(request.sid)
        prompt = build_booking_system_prompt(session)
        try:
            llm_response = get_llm_response(
                transcription, LLM_PROVIDER,
                openrouter_key=OPENROUTER_API_KEY, cohere_key=COHERE_API_KEY,
                system_message=prompt
            )
        except Exception:
            llm_response = "系统繁忙。"
        latency_info['llm_response'] = (time.time() - llm_start) * 1000

        session.add_to_history(transcription, llm_response)

        # 4. 统计与写入数据 (供 analyze_latency.py 读取)
        total_latency = (time.time() - start_time) * 1000

        import json
        structured_data = {
            'total': round(total_latency, 2),
            'conversion': 0,  # 已优化归零
            'vad': 0,  # 已优化归零
            'trim': 0,  # 已优化归零
            'asr': round(latency_info['asr_transcription'], 2),
            'llm': round(latency_info['llm_response'], 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open("stats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(structured_data) + "\n")

        # 5. 发送响应给前端
        emit('bot_response', {
            'user_text': transcription,
            'bot_text': llm_response,
            'latency_ms': {'backend': int(total_latency)}
        })

    except Exception as e:
        logger.error(f"Error: {e}")
        emit('error', {'message': str(e)})



if __name__ == '__main__':
    # allow_unsafe_werkzeug 允许在开发环境中使用 WebSocket
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)