#!/usr/bin/env python3
"""Test script for conversation detection system.

This script tests the conversation detection system by:
1. Capturing real audio input
2. Running it through VAD
3. Detecting conversations
4. Saving detected conversations to WAV files
"""

import logging
import time
from pathlib import Path
import threading
from queue import Queue
import datetime
import sys

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

import numpy as np
import sounddevice as sd

from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig
from src.conversation.detector import ConversationDetector
from src.visualization.vad import VADVisualizer

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def on_conversation(conversation):
    """Handle detected conversation."""
    # Create output directory
    output_dir = Path("debug/conversations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = output_dir / f"conversation_{timestamp}.wav"
    
    # Save conversation audio
    import wave
    with wave.open(str(wav_path), 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(44100)  # 44.1kHz
        # Convert float32 to int16
        audio_int16 = (conversation.audio_data * 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())
    
    logger.info(f"Saved conversation to {wav_path}")
    logger.info(f"Duration: {conversation.duration:.1f}s")
    logger.info(f"Speech segments: {len(conversation.metadata['speech_segments'])}")

def main():
    """Run conversation detection test."""
    # Initialize components
    vad_config = VADConfig(
        threshold=0.5,  # More aggressive speech detection
        min_conversation_duration_sec=3.0,  # Shorter for testing
        conversation_cooldown_sec=5.0  # Shorter cooldown for testing
    )
    
    vad = VADProcessor(config=vad_config)
    audio = AudioCapture()
    detector = ConversationDetector(
        audio_processor=audio,
        vad_processor=vad,
        buffer_duration_sec=10.0,
        pre_speech_sec=3.0,  # Capture 3s before first speech
        on_conversation=on_conversation
    )
    
    # Initialize visualization
    viz = VADVisualizer(
        window_size=8000,  # Show more samples
        samples_per_second=200,  # Faster update rate
        title="Conversation Detection Test"
    )
    
    # Audio callback
    def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray) -> None:
        """Process captured audio data."""
        try:
            # Get VAD state
            state = vad.get_state()
            
            # Update visualization
            viz.add_data(original, state.speech_probability, state.is_speech, state.is_conversation)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    try:
        # Register callback and start components
        audio.add_callback(handle_audio)
        detector.start()
        audio.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        # Clean up components in reverse order
        try:
            if viz and hasattr(viz, 'cleanup') and hasattr(viz, 'anim') and viz.anim and viz.anim.event_source:
                viz.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up visualization: {e}")
        
        try:
            audio.stop()
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
        
        try:
            vad.stop()
        except Exception as e:
            logger.error(f"Error stopping VAD: {e}")
        
        try:
            detector.stop()
        except Exception as e:
            logger.error(f"Error stopping detector: {e}")

if __name__ == "__main__":
    main()
