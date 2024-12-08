"""Test script for audio capture functionality."""

import logging
import socket
import time
import signal
import sys
from pathlib import Path

import sounddevice as sd
import numpy as np

# Add src to Python path
src_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(src_dir))

from src.audio.capture import AudioCapture

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG  # Changed to DEBUG for more verbose output
)
logger = logging.getLogger(__name__)

def analyze_audio_stream(host: str = "staple.local", port: int = 12345) -> None:
    """Analyze the raw audio stream format.
    
    Args:
        host: Hostname to test
        port: Port to test
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        # Get IP address
        ip = socket.gethostbyname(host)
        logger.debug(f"Resolved {host} to {ip}")
        
        # Try to connect
        sock.connect((host, port))
        logger.info(f"Successfully connected to {host}:{port}")
        
        # Try to read some initial data
        logger.debug("Reading initial data for analysis...")
        data = sock.recv(4096)
        if data:
            # Try to interpret the data in different ways
            logger.debug(f"Received {len(data)} bytes")
            
            # Look at raw bytes
            logger.debug(f"First 20 bytes: {list(data[:20])}")
            
            # Try as float32
            try:
                as_float32 = np.frombuffer(data, dtype=np.float32)
                logger.debug(f"As float32: min={as_float32.min():.3f}, max={as_float32.max():.3f}, mean={as_float32.mean():.3f}")
                logger.debug(f"First 5 float32 samples: {as_float32[:5]}")
                
                # Additional format checks
                if as_float32.min() >= -1.1 and as_float32.max() <= 1.1:
                    logger.info("Audio values are in valid float32 range (-1 to 1)")
                else:
                    logger.warning("Audio values outside normal float32 range!")
                    
                # Check for potential DC offset
                mean = as_float32.mean()
                if abs(mean) > 0.1:
                    logger.warning(f"Significant DC offset detected: {mean:.3f}")
                    
                # Check for potential silence or noise
                std = as_float32.std()
                if std < 0.01:
                    logger.warning("Very low audio variance - might be silence or DC")
                elif std > 0.5:
                    logger.warning("High audio variance - might be noise")
                    
            except Exception as e:
                logger.error(f"Failed to interpret as float32: {e}")
                
            # Try as int16
            try:
                as_int16 = np.frombuffer(data, dtype=np.int16)
                logger.debug(f"As int16: min={as_int16.min()}, max={as_int16.max()}, mean={as_int16.mean():.1f}")
            except:
                logger.debug("Failed to interpret as int16")
                
        else:
            logger.warning("No initial data received")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        sock.close()

def main():
    """Main test function."""
    # First analyze the stream format
    analyze_audio_stream()
    
    logger.info("Starting audio capture test...")
    
    # Initialize capture with correct format
    capture = AudioCapture(
        channels=1,  # Mono audio
        chunk_size=512,  # Match server's chunk size
        original_rate=44000,  # Match server's sample rate
        target_rate=16000  # Target rate for VAD
    )
    
    try:
        # Start capture
        capture.start()
        
        # Capture for 5 seconds
        logger.info("Capturing audio for 5 seconds...")
        time.sleep(5)
        
        # Get audio data
        original, downsampled = capture.get_audio_data()
        
        if len(original) > 0 and len(downsampled) > 0:
            logger.info(f"Captured {len(original)} original samples and {len(downsampled)} downsampled samples")
            logger.debug(f"Original - min: {original.min():.3f}, max: {original.max():.3f}, mean: {original.mean():.3f}, std: {original.std():.3f}")
            logger.debug(f"Downsampled - min: {downsampled.min():.3f}, max: {downsampled.max():.3f}, mean: {downsampled.mean():.3f}, std: {downsampled.std():.3f}")
            
            # Normalize if needed (shouldn't be necessary for float32 but just in case)
            if original.max() > 1.0 or original.min() < -1.0:
                logger.warning("Normalizing original audio...")
                original = np.clip(original, -1.0, 1.0)
            
            if downsampled.max() > 1.0 or downsampled.min() < -1.0:
                logger.warning("Normalizing downsampled audio...")
                downsampled = np.clip(downsampled, -1.0, 1.0)
            
            # Play back original audio
            logger.info("Playing back original audio...")
            sd.play(original, samplerate=44000)
            sd.wait()
            
            # Play back downsampled audio
            logger.info("Playing back downsampled audio...")
            sd.play(downsampled, samplerate=16000)
            sd.wait()
        else:
            logger.warning("No audio data captured")
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        # Clean up
        capture.stop()
        logger.info("Test complete")

if __name__ == "__main__":
    main()
