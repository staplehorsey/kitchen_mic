"""Audio capture module for Kitchen Mic.

Handles direct audio capture from microphone with dual-stream processing:
- Original 44kHz high-fidelity stream for storage
- Downsampled 16kHz stream for voice activity detection
"""

import logging
import threading
import time
import signal
from typing import Optional, Tuple, Callable

import numpy as np
import librosa
import pyaudio

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def visualize_audio_level(audio_chunk: np.ndarray, width: int = 50) -> str:
    """Create a visual representation of audio level.
    
    Args:
        audio_chunk: Audio samples
        width: Width of visualization
        
    Returns:
        ASCII visualization of audio level
    """
    # Calculate RMS level (root mean square)
    rms = np.sqrt(np.mean(np.square(audio_chunk)))
    
    # Apply log scaling and normalization
    # -60dB to 0dB range
    db = 20 * np.log10(max(rms, 1e-10))
    db = max(-60, min(0, db))  # Clamp to -60..0 dB
    level = (db + 60) / 60  # Normalize to 0..1
    
    # Create bar with multiple thresholds
    bar_width = int(level * width)
    bar = ""
    for i in range(width):
        if i < bar_width:
            if i / width < 0.2:  # First 20% - quiet
                bar += "="
            elif i / width < 0.6:  # 20-60% - normal speech
                bar += "█"
            else:  # 60-100% - loud
                bar += "▇"
        else:
            bar += " "
    
    # Add numeric level in dB
    return f"[{bar}] ({db:.1f}dB)"

class AudioCapture:
    """Captures audio directly from microphone and provides dual-stream processing."""
    
    def __init__(
        self,
        buffer_size: int = 4096,
        original_rate: int = 44000,  # Match server's sample rate
        target_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,  # Match server's chunk size
        visualize: bool = True,
        visualization_width: int = 50,
        visualization_interval: float = 0.1
    ):
        """Initialize audio capture.
        
        Args:
            buffer_size: Audio buffer size
            original_rate: Original sample rate (Hz)
            target_rate: Target sample rate for downsampling (Hz)
            channels: Number of audio channels
            chunk_size: Audio chunk size in samples
            visualize: Whether to show audio level visualization
            visualization_width: Width of visualization
            visualization_interval: Update interval for visualization
        """
        self.buffer_size = buffer_size
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.visualize = visualize
        self.visualization_width = visualization_width
        self.visualization_interval = visualization_interval
        
        logger.info(f"AudioCapture initialized with visualize={visualize}")
        
        # PyAudio state
        self.pyaudio = None
        self.stream = None
        self._running = False
        self.device_index = None
        
        # Audio format info (float32)
        self.format = pyaudio.paFloat32
        self.bytes_per_sample = channels * 4  # 4 bytes per float32
        
        # Audio level visualization
        self.last_level_time = 0
        self.level_interval = self.visualization_interval
        self.last_level = ""  # Store last level for cleanup
        
        logger.info(
            f"Audio format: {channels} channels, {original_rate}Hz, "
            f"chunk_size={chunk_size} samples"
        )
        
        # Callbacks for audio data
        self._callbacks = []
        self._lock = threading.Lock()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _find_microphone(self) -> Optional[int]:
        """Find Blue Yeti or default microphone."""
        # Try to find Blue Yeti by checking all devices
        for i in range(self.pyaudio.get_device_count()):
            dev_info = self.pyaudio.get_device_info_by_index(i)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Found audio device {i}: {dev_info['name']}")
            
            # Look for Blue Yeti in name and make sure it's an ALSA device
            if "Blue" in dev_info["name"] and "hw:" in dev_info["name"]:
                # Extract ALSA card and device numbers
                try:
                    hw_str = dev_info["name"].split("hw:")[1].split(")")[0]
                    card, device = map(int, hw_str.split(","))
                    logger.info(f"Found Blue Yeti on card {card}")
                    
                    # Test if device is actually available for capture
                    test_stream = self.pyaudio.open(
                        format=self.format,
                        channels=1,
                        rate=self.original_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024,
                        start=False
                    )
                    test_stream.close()
                    return i
                except Exception as e:
                    logger.debug(f"Failed to test Blue Yeti device: {e}")
                    continue
        
        # Fall back to default input device
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            logger.info(f"Using default microphone: {default_input['name']}")
            return default_input["index"]
        except Exception as e:
            logger.debug(f"Failed to get default input device: {e}")
            
            # Last resort: try to find any working input device
            for i in range(self.pyaudio.get_device_count()):
                dev_info = self.pyaudio.get_device_info_by_index(i)
                try:
                    test_stream = self.pyaudio.open(
                        format=self.format,
                        channels=1,
                        rate=self.original_rate,
                        input=True,
                        input_device_index=i,
                        frames_per_buffer=1024,
                        start=False
                    )
                    test_stream.close()
                    logger.info(f"Using microphone: {dev_info['name']}")
                    return i
                except:
                    continue
            
            raise ValueError("No working microphone found")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Handle incoming audio data from PyAudio."""
        if status:
            if status & pyaudio.paInputOverflow:
                logger.warning("Audio input overflow - some audio data was dropped")
            elif status & pyaudio.paInputUnderflow:
                logger.warning("Audio input underflow - gaps in audio data")
            else:
                logger.warning(f"PyAudio status: {status}")

        try:
            # Convert to numpy array (float32)
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            current_time = time.time()
            
            # Downsample for VAD
            downsampled = librosa.resample(
                audio_data,
                orig_sr=self.original_rate,
                target_sr=self.target_rate
            )
            
            # Visualize audio level if enabled
            if self.visualize:
                now = time.time()
                if now - self.last_level_time >= self.level_interval:
                    viz = visualize_audio_level(audio_data, width=self.visualization_width)
                    print(f"\r{' ' * len(self.last_level)}\r{viz}", end="", flush=True)
                    self.last_level = viz
                    self.last_level_time = now
            
            # Notify callbacks
            with self._lock:
                for callback in self._callbacks:
                    try:
                        callback(current_time, audio_data, downsampled)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")

        return (None, pyaudio.paContinue)

    def add_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Add callback for audio data.
        
        Args:
            callback: Function(timestamp, original_chunk, downsampled_chunk)
                timestamp: Current wall clock time
                original_chunk: Original audio data (44kHz)
                downsampled_chunk: Downsampled audio data (16kHz)
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Remove callback for audio data."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _cleanup(self) -> None:
        """Clean up PyAudio resources."""
        logger.debug("Cleaning up PyAudio resources")
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
        self.stream = None
        self.pyaudio = None
        
        if self.visualize:
            # Clear the audio level line when stopping
            print(f"\r{' ' * len(self.last_level)}\r", end="", flush=True)
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self) -> None:
        """Start audio capture."""
        if self._running:
            logger.warning("Audio capture already running")
            return
        
        logger.info("Starting audio capture")
        
        try:
            # Initialize PyAudio
            self.pyaudio = pyaudio.PyAudio()
            
            # Find microphone
            self.device_index = self._find_microphone()
            if self.device_index is None:
                raise ValueError("No microphone found")
            
            # Get device info
            info = self.pyaudio.get_device_info_by_index(self.device_index)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Selected device info:")
                logger.debug(f"  Name: {info['name']}")
                logger.debug(f"  Index: {self.device_index}")
                logger.debug(f"  Sample rate: {info['defaultSampleRate']}")
            
            # Use 1 channel since we've tested it works
            self.channels = 1
            
            # Open audio stream with larger buffer for stability
            frames_per_buffer = max(2048, self.chunk_size * 4)
            logger.debug(f"Using buffer size: {frames_per_buffer}")
            
            self.stream = self.pyaudio.open(
                format=self.format,
                channels=self.channels,
                rate=self.original_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=frames_per_buffer,
                stream_callback=self._audio_callback,
                start=False
            )
            
            # Start the stream
            self.stream.start_stream()
            logger.info("Audio capture started")
            
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            self._cleanup()
            self._running = False
            raise
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return
        
        logger.info("Stopping audio capture")
        self._running = False
        self._cleanup()
        logger.info("Audio capture stopped")
    
    @property
    def sample_rate(self) -> int:
        """Get the original sample rate."""
        return self.original_rate
