"""Test script for audio visualization."""

import logging
import sys
from pathlib import Path
import time
import threading
from queue import Queue

import numpy as np
import sounddevice as sd

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.player import AudioPlayer
from src.visualization.audio import AudioVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_audio(duration=10.0, sample_rate=44100):
    """Generate test audio signal.
    
    Creates a mix of:
    - 440 Hz sine (A4 note)
    - 880 Hz sine (A5 note)
    - White noise
    - Fade in/out envelope
    """
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Generate components
    sine1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    sine2 = 0.3 * np.sin(2 * np.pi * 880 * t)
    noise = 0.1 * np.random.randn(len(t))
    
    # Create fade envelope
    fade_time = 0.1  # seconds
    fade_samples = int(fade_time * sample_rate)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    envelope = np.ones(len(t))
    envelope[:fade_samples] = fade_in
    envelope[-fade_samples:] = fade_out
    
    # Combine and normalize
    audio = (sine1 + sine2 + noise) * envelope
    audio = audio / np.max(np.abs(audio))
    
    return audio

def main():
    """Run audio visualization test."""
    try:
        # Generate test audio
        sample_rate = 44100
        audio = generate_test_audio(duration=10.0, sample_rate=sample_rate)
        chunk_size = 512
        
        # Create audio player
        player = AudioPlayer(sample_rate=sample_rate, chunk_size=chunk_size)
        
        # Create visualizer
        visualizer = AudioVisualizer(
            window_size=2000,
            samples_per_second=50,
            title="Audio Test"
        )
        visualizer.audio_player = player
        
        # Process audio in chunks
        def process_audio():
            num_chunks = len(audio) // chunk_size
            for i in range(num_chunks):
                if not visualizer.running:
                    break
                
                # Get chunk
                start = i * chunk_size
                end = start + chunk_size
                chunk = audio[start:end]
                
                # Add to player and visualizer
                player.play(chunk)
                visualizer.add_audio(chunk)
                
                # Control playback rate
                time.sleep(chunk_size / sample_rate)
        
        # Start audio processing thread
        audio_thread = threading.Thread(target=process_audio)
        audio_thread.start()
        
        # Start visualization (blocks until window closed)
        visualizer.start_animation()
        visualizer.show()
        
        # Cleanup
        player.stop()
        audio_thread.join()
        
    except Exception as e:
        logger.error(f"Error in test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
