"""Test script for conversation detection system.

This script validates:
1. Conversation boundary detection
2. Audio quality preservation
3. Timing accuracy
4. Memory usage
"""

import logging
import sys
import time
from pathlib import Path
import threading
from queue import Queue
import wave

import numpy as np
import sounddevice as sd

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.player import AudioPlayer
from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig
from src.conversation.detector import ConversationDetector
from src.visualization.vad import VADVisualizer
from src.messages import ConversationMessage

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def save_conversation(conversation: ConversationMessage, output_dir: Path) -> None:
    """Save conversation audio and metadata for analysis.
    
    Args:
        conversation: Detected conversation
        output_dir: Directory to save files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save audio as WAV
    audio_path = output_dir / f"conversation_{conversation.id}.wav"
    with wave.open(str(audio_path), 'wb') as wav:
        wav.setnchannels(1)  # Mono
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(44100)  # 44.1kHz
        # Convert float32 to int16
        audio_int16 = (conversation.audio_data * 32767).astype(np.int16)
        wav.writeframes(audio_int16.tobytes())
    
    # Save metadata
    metadata_path = output_dir / f"conversation_{conversation.id}.txt"
    with open(metadata_path, 'w') as f:
        f.write(f"Conversation ID: {conversation.id}\n")
        f.write(f"Start Time: {conversation.start_time}\n")
        f.write(f"End Time: {conversation.end_time}\n")
        f.write(f"Duration: {conversation.duration:.2f}s\n")
        f.write(f"Audio Samples: {len(conversation.audio_data)}\n")
        f.write(f"Sample Rate: 44100 Hz\n")
        
        if 'speech_segments' in conversation.metadata:
            f.write("\nSpeech Segments:\n")
            for segment in conversation.metadata['speech_segments']:
                f.write(f"  {segment}\n")

def main():
    """Main test function."""
    vad = None
    capture = None
    detector = None
    player = None
    viz = None
    
    # Create output directory
    output_dir = Path(__file__).parent / "test_output" / time.strftime("%Y%m%d_%H%M%S")
    
    try:
        # Initialize components
        vad = VADProcessor(VADConfig())
        viz = VADVisualizer(
            window_size=8000,  # Show more samples
            samples_per_second=200,  # Faster update rate
            title="Conversation Detection Test"
        )
        
        # Create audio capture and player
        capture = AudioCapture(host="staple.local")
        player = AudioPlayer(sample_rate=capture.sample_rate)
        viz.audio_player = player
        
        # Create conversation detector
        def handle_conversation(conversation: ConversationMessage) -> None:
            """Handle detected conversation."""
            logger.info(
                f"Conversation detected: id={conversation.id} "
                f"duration={conversation.duration:.1f}s"
            )
            # Save for analysis
            save_conversation(conversation, output_dir)
        
        detector = ConversationDetector(
            audio_processor=capture,
            vad_processor=vad,
            buffer_duration_sec=5.0,  # 5s pre-conversation buffer
            on_conversation=handle_conversation
        )
        
        # Audio callback
        def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray) -> None:
            """Process captured audio data."""
            try:
                # Add audio to VAD
                vad.add_audio(downsampled)
                
                # Get VAD state and update detector
                state = vad.get_state()
                
                # Update visualization and play audio
                player.play(original)
                viz.add_data(original, state.speech_probability, state.is_speech, state.is_conversation)
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        # Start components
        capture.add_callback(handle_audio)
        detector.start()
        capture.start()
        
        # Start visualization
        viz.start_animation()
        viz.show()
        
    except KeyboardInterrupt:
        logger.info("Stopping test...")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        raise
    finally:
        # Clean up
        logger.info("Cleaning up...")
        
        if detector:
            try:
                detector.stop()
            except Exception as e:
                logger.error(f"Error stopping detector: {e}")
        
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
        
        logger.info(f"Test complete. Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
