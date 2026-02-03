"""
Voice Activity Detection utilities using Silero VAD
Detects speech in audio and trims silence
"""
import logging
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from pydub import AudioSegment
import os

logger = logging.getLogger(__name__)

# Global VAD model (loaded once at startup)
vad_model = None


def initialize_vad():
    """Load Silero VAD model once at startup"""
    global vad_model
    try:
        logger.info("Loading Silero VAD model...")
        torch.set_num_threads(1)  # For efficiency on CPU
        vad_model = load_silero_vad()
        logger.info("✓ Silero VAD model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load VAD model: {e}")
        return False


def validate_speech(wav_path, min_speech_duration_ms=250):
    """
    Check if audio contains speech

    Args:
        wav_path: Path to WAV file
        min_speech_duration_ms: Minimum speech duration to consider valid (default 250ms)

    Returns:
        (has_speech: bool, speech_duration_ms: float)
    """
    global vad_model

    if vad_model is None:
        logger.warning("VAD model not initialized, skipping validation")
        return True, 0  # Assume speech if VAD not available

    try:
        # Read audio
        wav = read_audio(wav_path)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=True
        )

        if not speech_timestamps:
            logger.info(f"No speech detected in {wav_path}")
            return False, 0

        # Calculate total speech duration
        total_speech_duration = sum(
            (segment['end'] - segment['start']) * 1000  # Convert to ms
            for segment in speech_timestamps
        )

        has_speech = total_speech_duration >= min_speech_duration_ms

        logger.info(f"Speech validation: {has_speech} (duration: {total_speech_duration:.2f}ms)")
        return has_speech, total_speech_duration

    except Exception as e:
        logger.error(f"Error validating speech: {e}")
        return True, 0  # Fail open - assume speech if error


def trim_silence(wav_path, output_path=None):
    """
    Trim silence from audio using VAD

    Args:
        wav_path: Path to input WAV file
        output_path: Path to output WAV file (optional, defaults to temp file)

    Returns:
        (success: bool, trimmed_path: str, duration_saved_ms: float)
    """
    global vad_model

    if vad_model is None:
        logger.warning("VAD model not initialized, skipping trimming")
        return False, wav_path, 0

    try:
        # Read audio
        wav = read_audio(wav_path)

        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            wav,
            vad_model,
            return_seconds=True
        )

        if not speech_timestamps:
            logger.warning(f"No speech detected in {wav_path}, cannot trim")
            return False, wav_path, 0

        # Load audio with pydub
        audio = AudioSegment.from_wav(wav_path)
        original_duration = len(audio)

        # Extract speech segments
        speech_segments = []
        for segment in speech_timestamps:
            start_ms = int(segment['start'] * 1000)
            end_ms = int(segment['end'] * 1000)
            speech_segments.append(audio[start_ms:end_ms])

        # Combine all speech segments
        trimmed_audio = sum(speech_segments) if len(speech_segments) > 1 else speech_segments[0]

        # Generate output path if not provided
        if output_path is None:
            output_path = wav_path.replace('.wav', '_trimmed.wav')

        # Export trimmed audio
        trimmed_audio.export(output_path, format='wav')

        duration_saved = original_duration - len(trimmed_audio)
        logger.info(f"Trimmed silence: saved {duration_saved}ms ({duration_saved/original_duration*100:.1f}%)")

        return True, output_path, duration_saved

    except Exception as e:
        logger.error(f"Error trimming silence: {e}")
        return False, wav_path, 0


def get_speech_probability(wav_path):
    """
    Get speech probability score for audio

    Args:
        wav_path: Path to WAV file

    Returns:
        float: Speech probability (0.0 to 1.0)
    """
    global vad_model

    if vad_model is None:
        return 1.0  # Assume speech if VAD not available

    try:
        wav = read_audio(wav_path)
        speech_timestamps = get_speech_timestamps(wav, vad_model, return_seconds=True)

        if not speech_timestamps:
            return 0.0

        # Calculate ratio of speech to total duration
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio) / 1000.0  # Convert to seconds

        speech_duration = sum(
            segment['end'] - segment['start']
            for segment in speech_timestamps
        )

        return min(speech_duration / total_duration, 1.0)

    except Exception as e:
        logger.error(f"Error calculating speech probability: {e}")
        return 1.0
