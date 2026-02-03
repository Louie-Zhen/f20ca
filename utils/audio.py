"""
Audio processing utilities for Garage Booking Assistant
Handles audio format conversion
"""
import os
import tempfile
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_webm_to_wav(webm_data):
    """Convert WebM audio to WAV format"""
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file:
            webm_file.write(webm_data)
            webm_path = webm_file.name

        wav_path = webm_path.replace('.webm', '.wav')

        # Convert using pydub
        audio = AudioSegment.from_file(webm_path, format="webm")

        # Convert to mono, 16kHz (required by most ASR models)
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        # Export as WAV
        audio.export(wav_path, format="wav")

        # Clean up webm file
        os.remove(webm_path)

        return wav_path

    except Exception as e:
        logger.error(f"Error converting audio: {e}")
        raise


def combine_audio_files(user_wav_path, bot_wav_path, session_id, recordings_dir, add_silence=True):
    """Combine user and bot audio into a single WAV file with optional silence between them"""
    try:
        # Load both audio files
        user_audio = AudioSegment.from_wav(user_wav_path)
        bot_audio = AudioSegment.from_wav(bot_wav_path)

        # Add 500ms of silence between user and bot audio
        if add_silence:
            silence = AudioSegment.silent(duration=500)  # 500ms
            combined_audio = user_audio + silence + bot_audio
        else:
            combined_audio = user_audio + bot_audio

        # Save combined audio
        combined_audio_dir = os.path.join(recordings_dir, 'combined_audio')
        dest_path = os.path.join(combined_audio_dir, f'{session_id}_combined.wav')
        combined_audio.export(dest_path, format='wav')

        logger.info(f"Saved combined audio: {dest_path}")
        return dest_path
    except Exception as e:
        logger.error(f"Error combining audio files: {e}")
        return None
