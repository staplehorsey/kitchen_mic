#!/usr/bin/env python3
"""Test script for conversation orchestration.

This script tests the complete conversation pipeline by:
1. Capturing real audio input
2. Running VAD with visualization
3. Processing detected conversations
4. Saving results with proper organization
"""

import logging
import sys
import time
from pathlib import Path
import numpy as np

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.conversation.orchestrator import ConversationOrchestrator
from src.conversation.detector import ConversationDetector
from src.audio.capture import AudioCapture
from src.vad.processor import VADProcessor, VADConfig
from src.visualization.vad import VADVisualizer
from src.storage.persistence import ConversationStorage
from src.transcription.processor import TranscriptionProcessor
from src.llm.processor import SummaryProcessor

# Configure logging - only show INFO and above for most modules
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
# Reduce noise from some modules
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('whisper').setLevel(logging.WARNING)
logging.getLogger('src.audio.capture').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def main():
    """Run orchestrator test with real audio."""
    # Create components
    storage = ConversationStorage(Path("debug/storage"))
    transcription = TranscriptionProcessor(model_name="base")
    summary = SummaryProcessor()
    
    # Initialize audio and VAD
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
        pre_speech_sec=3.0  # Capture 3s before first speech
    )
    
    # Create orchestrator with components
    orchestrator = ConversationOrchestrator(
        detector=detector,
        storage=storage,
        transcription_processor=transcription,
        summary_processor=summary,
        max_queue_size=10
    )
    
    # Initialize visualization
    viz = VADVisualizer(
        window_size=8000,  # Show more samples
        samples_per_second=200,  # Faster update rate
        title="Conversation Pipeline Test"
    )
    
    # Audio callback for visualization
    def handle_audio(timestamp: float, original: np.ndarray, downsampled: np.ndarray) -> None:
        """Process captured audio data."""
        try:
            # Get VAD state
            state = vad.get_state()
            
            # Update visualization
            viz.add_data(original, state.speech_probability, state.is_speech, state.is_conversation)
            
            # Log orchestrator status periodically
            if timestamp % 10 < 0.1:  # Every ~10 seconds
                status = orchestrator.get_status()
                if status['queue_size'] > 0:
                    logger.info(f"Queue size: {status['queue_size']}")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
    
    try:
        # Register callback and start components
        audio.add_callback(handle_audio)
        orchestrator.start()  # This starts the detector too
        audio.start()
        
        logger.info("Started pipeline - speak to test! Press Ctrl+C to stop")
        
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
            orchestrator.stop()  # This stops the detector too
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
        
        try:
            audio.stop()
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
        
        try:
            vad.stop()
        except Exception as e:
            logger.error(f"Error stopping VAD: {e}")

if __name__ == "__main__":
    main()
