from flask import Flask, render_template, send_from_directory, request
from flask_socketio import SocketIO, emit
import base64
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import requests
from flask import request, Response, stream_with_context
import shutil
import time
from elevenlabs.client import ElevenLabs
from io import BytesIO

# Import utilities
from utils.llm import get_llm_response, build_booking_system_prompt
from utils.audio import convert_webm_to_wav, combine_audio_files
from utils.recording import save_recording_metadata
from utils.booking_state import get_or_create_session
from utils.calendar import initialize_calendar
from utils.vad import initialize_vad, validate_speech, trim_silence

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Flask app
app = Flask(__name__,
            static_folder='.',
            template_folder='.')
app.config['SECRET_KEY'] = 'garage-booking-secret'

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ElevenLabs Client for STT
# TTS is handled by frontend using ElevenLabs via Puter.js
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
if not ELEVENLABS_API_KEY:
    logger.error("ELEVENLABS_API_KEY not found in environment variables!")
    raise ValueError("Please set ELEVENLABS_API_KEY in .env file")

elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

# LLM Provider Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'cohere').lower()  # Default to cohere
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Validate configuration
if LLM_PROVIDER == 'openrouter' and not OPENROUTER_API_KEY:
    logger.error("OPENROUTER_API_KEY not found in environment variables!")
    raise ValueError("Please set OPENROUTER_API_KEY in .env file")

if LLM_PROVIDER == 'cohere' and not COHERE_API_KEY:
    logger.error("COHERE_API_KEY not found in environment variables!")
    raise ValueError("Please set COHERE_API_KEY in .env file")

logger.info(f"LLM Provider: {LLM_PROVIDER.upper()}")
logger.info("STT Provider: ELEVENLABS")

# Recording directories
RECORDINGS_DIR = 'recordings'
COMBINED_AUDIO_DIR = os.path.join(RECORDINGS_DIR, 'combined_audio')
METADATA_DIR = os.path.join(RECORDINGS_DIR, 'metadata')

# Ensure recording directories exist
for directory in [COMBINED_AUDIO_DIR, METADATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# Initialize calendar
initialize_calendar()

# Initialize VAD model
initialize_vad()

# Global latency tracking
latency_records = []

# LLM functions moved to utils/llm.py

# STT is now handled by ElevenLabs API - no model loading needed!
# TTS is handled by frontend using ElevenLabs via Puter.js
# No heavyweight models to load at startup!

# Audio and recording functions moved to utils/audio.py and utils/recording.py

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index-zh.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files (CSS, JS, JSON)"""
    return send_from_directory('.', path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected")
    emit('status', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected")

# 新增的 TTS 代理接口
# 修改：支持 GET 方法，从 URL 参数获取文字
@app.route('/api/tts', methods=['GET', 'POST'])
def tts_proxy():
    """安全语音代理 (极速转发版 - 专门适配短对话)"""
    if request.method == 'GET':
        text = request.args.get('text', '')
    else:
        text = request.json.get('text', '')

    url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM?output_format=mp3_44100_128"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY # 安全读取 .env
    }
    payload = {
        "text": text,
        "model_id": "eleven_flash_v2_5"
    }

    # 1. 后端一口气向 ElevenLabs 获取完整的 MP3 (因为只有十几字，耗时极短)
    # 注意这里去掉了 stream=True
    response = requests.post(url, json=payload, headers=headers)

    # 2. 一口气完整转发给前端，浏览器收到后会瞬间开始完美播放
    return Response(
        response.content,
        content_type='audio/mpeg'
    )


@socketio.on('audio_data')
def handle_audio_data(data):
    """Handle incoming audio data from frontend (Direct WebM Version)"""
    user_audio_path = None
    try:
        # Start total timing
        start_time = time.time()

        # 因为跳过了转码和本地VAD，这些延迟统统归零
        latency_info = {
            'audio_conversion': 0.0,
            'vad_validation': 0.0,
            'silence_trimming': 0.0,
            'asr_transcription': 0,
            'llm_response': 0
        }

        logger.info("Received audio data")

        # Check if recording mode is enabled
        recording_mode = data.get('recording_mode', False)
        session_id = None
        timestamp = None

        if recording_mode:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            session_id = f"session_{timestamp}"
            logger.info(f"Recording mode enabled - Session ID: {session_id}")

        # Decode base64 audio directly to WebM bytes (NO WAV CONVERSION)
        audio_base64 = data.get('audio')
        if not audio_base64:
            emit('error', {'message': 'No audio data received'})
            return

        audio_bytes = base64.b64decode(audio_base64)
        logger.info(f"Decoded WebM audio: {len(audio_bytes)} bytes")

        # [OPTIONAL] Save temporary copy for recording history if enabled
        if recording_mode and session_id:
            user_audio_path = os.path.join(COMBINED_AUDIO_DIR, f"{session_id}_user.webm")
            with open(user_audio_path, 'wb') as f:
                f.write(audio_bytes)

        # -----------------------------------------------------------------
        # 1. Transcribe audio using ElevenLabs STT (DIRECT WebM)
        # -----------------------------------------------------------------
        logger.info("Transcribing WebM audio directly with ElevenLabs...")
        asr_start = time.time()

        # Wrap bytes in BytesIO
        audio_buffer = BytesIO(audio_bytes)

        # Call ElevenLabs STT API (Crucial: Use tuple to specify .webm MIME type)
        transcription_result = elevenlabs_client.speech_to_text.convert(
            file=("audio.webm", audio_buffer, "audio/webm"),
            model_id="scribe_v2",
            language_code="eng"  # zho中文
        )

        transcription = transcription_result.text
        latency_info['asr_transcription'] = (time.time() - asr_start) * 1000
        logger.info(f"Transcription: {transcription} ({latency_info['asr_transcription']:.2f}ms)")

        # Handle empty transcription
        if not transcription or transcription.strip() == "":
            logger.warning("Empty transcription received, skipping LLM call")
            emit('error', {'message': 'Could not understand audio. Please try again.'})
            if user_audio_path and os.path.exists(user_audio_path):
                os.remove(user_audio_path)
            return

        # -----------------------------------------------------------------
        # 2. Get LLM response
        # -----------------------------------------------------------------
        session_id_booking = request.sid
        booking_session = get_or_create_session(session_id_booking)
        system_prompt = build_booking_system_prompt(booking_session)

        llm_start = time.time()
        llm_response = get_llm_response(
            transcription,
            LLM_PROVIDER,
            openrouter_key=OPENROUTER_API_KEY,
            cohere_key=COHERE_API_KEY,
            groq_key=GROQ_API_KEY,
            system_message=system_prompt

        )
        latency_info['llm_response'] = (time.time() - llm_start) * 1000
        logger.info(f"LLM response received ({latency_info['llm_response']:.2f}ms)")

        booking_session.add_to_history(transcription, llm_response)

        # -----------------------------------------------------------------
        # 3. Calculate Latency & Write to stats.jsonl (Your Code)
        # -----------------------------------------------------------------
        backend_latency = (time.time() - start_time) * 1000
        logger.info(f"Total backend latency: {backend_latency:.2f}ms")

        # 🔴 新增：直接把数据写入专用的数据文件 stats.jsonl (结构化日志)
        import json
        structured_data = {
            'total': round(backend_latency, 2),
            'conversion': round(latency_info['audio_conversion'], 2),
            'vad': round(latency_info['vad_validation'], 2),
            'trim': round(latency_info['silence_trimming'], 2),
            'asr': round(latency_info['asr_transcription'], 2),
            'llm': round(latency_info['llm_response'], 2),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # 使用 'a' 模式追加写入，一行一个 JSON
        with open("stats.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(structured_data) + "\n")
        # ============================================================

        # Handle recording metadata saving
        avg_latency = None
        if recording_mode and session_id and user_audio_path:
            avg_latency = save_recording_metadata(
                session_id, transcription, llm_response, timestamp,
                latency_info, METADATA_DIR, latency_records
            )
            # 录音元数据保存后，删除临时文件
            os.remove(user_audio_path)

        # -----------------------------------------------------------------
        # 4. Emit Response
        # -----------------------------------------------------------------
        emit('bot_response', {
            'user_text': transcription,
            'bot_text': llm_response,
            'success': True,
            'recorded': recording_mode,
            'latency_ms': {
                'backend': round(backend_latency, 2),
                'average': round(avg_latency, 2) if avg_latency else None
            }
        })

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit('error', {'message': f'Transcription failed: {str(e)}'})
        if user_audio_path and os.path.exists(user_audio_path):
            try:
                os.remove(user_audio_path)
            except:
                pass

if __name__ == '__main__':
    logger.info("Starting Garage Booking Assistant server...")
    logger.info("=" * 60)
    logger.info("Loading models...")
    logger.info("VAD: Silero VAD (local)")
    logger.info("STT: ElevenLabs API")
    logger.info("TTS: ElevenLabs (frontend)")
    logger.info(f"LLM: {LLM_PROVIDER.upper()}")
    logger.info("=" * 60)
    logger.info("Server running on http://localhost:5001")
    logger.info("Ready for requests!")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)
