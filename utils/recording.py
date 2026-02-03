"""
Recording and metadata utilities for Garage Booking Assistant
Handles saving recordings and metadata to disk
"""
import os
import json
import logging

logger = logging.getLogger(__name__)


def save_recording_metadata(session_id, user_text, bot_text, timestamp, latency_info, metadata_dir, latency_records):
    """Save metadata for a recording session with latency information"""
    try:
        metadata_path = os.path.join(metadata_dir, f'{session_id}.json')

        # Calculate total latency
        total_latency = sum(latency_info.values())

        # Add to global latency tracking
        latency_records.append(total_latency)

        # Calculate average latency
        avg_latency = sum(latency_records) / len(latency_records)

        metadata = {
            'session_id': session_id,
            'timestamp': timestamp,
            'user_text': user_text,
            'bot_text': bot_text,
            'audio_file': f'{session_id}_combined.wav',
            'latency_ms': {
                'audio_conversion': round(latency_info.get('audio_conversion', 0), 2),
                'vad_validation': round(latency_info.get('vad_validation', 0), 2),
                'silence_trimming': round(latency_info.get('silence_trimming', 0), 2),
                'asr_transcription': round(latency_info.get('asr_transcription', 0), 2),
                'llm_response': round(latency_info.get('llm_response', 0), 2),
                'tts_generation': round(latency_info.get('tts_generation', 0), 2),
                'total': round(total_latency, 2)
            },
            'average_latency_ms': round(avg_latency, 2),
            'session_count': len(latency_records)
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata: {metadata_path}")
        logger.info(f"Total latency: {total_latency:.2f}ms | Average: {avg_latency:.2f}ms")
        return avg_latency
    except Exception as e:
        logger.error(f"Error saving metadata: {e}")
        return None
