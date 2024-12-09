"""
Transcription processor using Whisper for speech-to-text conversion.
Handles model configuration and error handling.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import whisper
import torch

from ..messages import ConversationMessage, TranscriptionMessage

# Configure logging
logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """Manages Whisper model and provides transcription processing."""
    
    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        """Initialize transcription processor.
        
        Args:
            model_name: Whisper model name to load
            device: Device to use ("cpu" or "cuda"), defaults to auto-detect
        """
        # Set device
        if device:
            self.device = device
            if device == "cpu":
                # Force CPU before loading model
                torch.cuda.is_available = lambda: False
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Loading Whisper model '{model_name}' on {self.device}")
        
        try:
            self.model = whisper.load_model(model_name)
            if self.device == "cpu":
                self.model = self.model.cpu()
            else:
                self.model = self.model.to(self.device)
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
        
        logger.debug("Transcription processor initialized")
    
    def _clear_cuda_cache(self) -> None:
        """Clear CUDA cache if using GPU."""
        if self.device == "cuda":
            # Clear cache and collect garbage
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def process_conversation(
        self,
        conversation: ConversationMessage,
        language_hint: str = "en"
    ) -> TranscriptionMessage:
        """Process a conversation and return transcription message.
        
        Args:
            conversation: Conversation to transcribe
            language_hint: Optional language hint for Whisper (e.g. "en", "es")
            
        Returns:
            TranscriptionMessage with results
        """
        start_time = time.time()
        logger.debug(f"Processing {conversation.duration:.1f}s of audio")
        
        try:
            # Clear CUDA cache before processing
            self._clear_cuda_cache()
            
            # Get 16kHz audio data
            audio = conversation.audio_data_16k
            
            # Ensure audio is float32 in [-1, 1]
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            if audio.max() > 1.0:
                audio = audio / 32768.0  # Convert from int16
            
            # Run transcription with improved options for reducing repetition
            result = self.model.transcribe(
                audio,
                language=language_hint,
                initial_prompt="This is a conversation in a kitchen.",
                task="transcribe",
                best_of=5,  # Use 5 beams for better quality
                beam_size=5,  # Match beam size to best_of
                patience=2.0,  # Higher patience for better beam search
                temperature=0.0,  # Reduce randomness
                compression_ratio_threshold=2.8,  # Higher threshold to reject repetitive text
                condition_on_previous_text=False,  # Don't condition on previous text to reduce repetition
                max_context=64,  # Reduced context size to prevent repetition
                fp16=False if self.device == "cpu" else torch.cuda.is_available()  # Use FP16 only if using GPU
            )
            
            # Create transcription message
            message = TranscriptionMessage.from_conversation(
                conversation,
                transcription={
                    'text': result['text'],
                    'segments': result['segments'],
                    'language': result['language'],
                    'timing': {
                        'process_start': start_time,
                        'process_end': time.time()
                    }
                }
            )
            
            # Clear CUDA cache after processing
            self._clear_cuda_cache()
            
            logger.info(
                f"Transcribed {conversation.id}: "
                f"{len(result['text'].split())} words"
            )
            return message
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
