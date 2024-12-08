#!/usr/bin/env python3
"""Test script for transcription system.

This script tests the transcription system by:
1. Loading conversation WAV files (44kHz and 16kHz)
2. Running them through Whisper
3. Saving transcription results
"""

import logging
import time
from pathlib import Path
import sys
import wave
import json

# Add src directory to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

import numpy as np

from src.messages import ConversationMessage
from src.transcription.processor import TranscriptionProcessor

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_wav(wav_path: Path) -> np.ndarray:
    """Load WAV file into numpy array.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Audio data as float32 array
    """
    with wave.open(str(wav_path), 'rb') as wav:
        # Get WAV info
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()
        
        # Read audio data
        audio_data = wav.readframes(n_frames)
        
        # Convert to numpy array
        dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
        audio = np.frombuffer(audio_data, dtype=dtype)
        
        # Convert to float32 [-1, 1]
        audio = audio.astype(np.float32) / np.iinfo(dtype).max
        
        # Convert to mono if stereo
        if channels == 2:
            audio = audio.reshape(-1, 2).mean(axis=1)
            
        return audio

def main():
    """Run transcription test."""
    # Get latest conversation file
    conv_dir = Path("debug/conversations")
    if not conv_dir.exists():
        logger.error("No conversations directory found")
        return
    
    # Find main WAV files (exclude _16k files)
    wav_files = [f for f in conv_dir.glob("*.wav") if not f.stem.endswith("_16k")]
    if not wav_files:
        logger.error("No WAV files found")
        return
    
    # Get latest WAV file and related files
    wav_path = max(wav_files, key=lambda p: p.stat().st_mtime)
    base_stem = wav_path.stem  # e.g. "conversation_20241208_162257"
    
    wav_16k_path = wav_path.parent / f"{base_stem}_16k.wav"
    json_path = wav_path.parent / f"{base_stem}.json"
    
    # Check all files exist
    if not wav_16k_path.exists():
        logger.error(f"Missing 16kHz audio file: {wav_16k_path}")
        return
        
    if not json_path.exists():
        logger.error(f"Missing metadata file: {json_path}")
        return
        
    logger.info(f"Processing conversation {base_stem}")
    
    # Load audio data
    audio_data = load_wav(wav_path)
    audio_data_16k = load_wav(wav_16k_path)
    
    # Load metadata
    with open(json_path) as f:
        metadata = json.load(f)
    
    # Create conversation message
    conversation = ConversationMessage(
        id=metadata["id"],
        start_time=metadata["start_time"],
        end_time=metadata["end_time"],
        audio_data=audio_data,  # Original 44kHz
        audio_data_16k=audio_data_16k,  # 16kHz version
        metadata=metadata
    )
    
    # Initialize transcription processor with large model
    processor = TranscriptionProcessor(model_name="large")
    
    # Process conversation
    start_time = time.time()
    transcription = processor.process_conversation(conversation)
    processing_time = time.time() - start_time
    
    # Save transcription
    output_dir = Path("debug/transcriptions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{base_stem}.json"
    with open(output_path, "w") as f:
        json.dump(transcription.to_dict(), f, indent=2)
    
    logger.info(f"Saved transcription to {output_path}")
    logger.info(f"Transcribed text: {transcription.transcription['text']}")
    logger.info(f"Processing time: {processing_time:.2f}s")

if __name__ == "__main__":
    main()
