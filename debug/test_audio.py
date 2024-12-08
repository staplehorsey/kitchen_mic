"""Test script for audio utilities."""

import sys
import time
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.player import AudioPlayer
from src.audio.utils import TimeRange, chunk_audio, resample_audio

def test_player():
    """Test audio player with sine wave."""
    try:
        # Create player
        sample_rate = 44000
        player = AudioPlayer(sample_rate=sample_rate)
        
        # Generate 5 seconds of 440Hz sine wave
        duration = 5
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Play in chunks
        chunk_size = 512
        chunks = chunk_audio(audio, chunk_size)
        
        print("Playing sine wave... Press Ctrl+C to stop")
        for chunk in chunks:
            player.play(chunk)
            time.sleep(chunk_size / sample_rate)  # Simulate real-time
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    
    finally:
        if player:
            player.stop()

def test_utils():
    """Test audio utilities."""
    # Test TimeRange
    tr = TimeRange(1.0, 2.0)
    assert tr.contains(1.5)
    assert not tr.contains(0.5)
    assert tr.duration == 1.0
    print("TimeRange tests passed")
    
    # Test chunk_audio
    audio = np.ones(1000)
    chunks = chunk_audio(audio, 256)
    assert len(chunks) == 4
    assert all(len(chunk) == 256 for chunk in chunks)
    print("chunk_audio tests passed")
    
    # Test resample_audio
    orig_rate = 44000
    target_rate = 16000
    duration = 1.0
    samples = int(orig_rate * duration)
    audio = np.random.rand(samples)
    
    resampled, dur = resample_audio(audio, orig_rate, target_rate)
    assert len(resampled) == int(target_rate * duration)
    assert abs(dur - duration) < 0.1
    print("resample_audio tests passed")

def main():
    """Main test function."""
    try:
        print("Testing audio utilities...")
        test_utils()
        
        print("\nTesting audio player...")
        test_player()
        
    except Exception as e:
        print(f"Error in tests: {e}")

if __name__ == "__main__":
    main()
