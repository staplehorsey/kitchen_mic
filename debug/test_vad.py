"""Test script for Voice Activity Detection."""

import logging
import sys
import time
from pathlib import Path
import threading
from queue import Queue

import numpy as np
import sounddevice as sd

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.player import AudioPlayer
from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig, VADState
from src.visualization.vad import VADVisualizer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    vad = None
    capture = None
    player = None
    viz = None
    
    try:
        # Initialize components
        vad = VADProcessor(VADConfig())
        viz = VADVisualizer(
            window_size=8000,  # Show more samples
            samples_per_second=200,  # Faster update rate
            title="Voice Activity Detection Test"
        )
        
        # Create audio capture first to get sample rate
        capture = AudioCapture(host="staple.local")
        player = AudioPlayer(sample_rate=capture.sample_rate)
        viz.audio_player = player
        
        # Audio callback
        def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray) -> None:
            """Process captured audio data."""
            try:
                # Add audio to VAD
                vad.add_audio(downsampled)
                
                # Get VAD state
                state = vad.get_state()
                
                # Update visualization and play audio
                player.play(original)
                viz.add_data(original, state.speech_probability, state.is_speech, state.is_conversation)
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        # Register callback and start capture
        capture.add_callback(handle_audio)
        capture.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up...")
        # Clean up components
        if capture:
            try:
                capture.stop()
            except Exception as e:
                logger.error(f"Error stopping capture: {e}")
        
        if viz:
            try:
                viz.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up viz: {e}")
        
        if player:
            try:
                player.stop()
            except Exception as e:
                logger.error(f"Error stopping player: {e}")

if __name__ == "__main__":
    main()
