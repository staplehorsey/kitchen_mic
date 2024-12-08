"""Audio capture module for Kitchen Mic.

Handles raw audio capture from network stream with dual-stream processing:
- Original 44kHz high-fidelity stream for storage
- Downsampled 16kHz stream for voice activity detection
"""

import logging
import socket
import threading
import time
import signal
from typing import Optional, Tuple, Callable
import struct
import errno

import numpy as np
import librosa

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AudioCapture:
    """Captures audio from network stream and maintains dual-stream processing."""
    
    def __init__(
        self,
        host: str = "staple.local",
        port: int = 12345,
        buffer_size: int = 4096,
        original_rate: int = 44000,  # Match server's sample rate
        target_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,  # Match server's chunk size
        on_data: Optional[Callable[[np.ndarray, np.ndarray], None]] = None  # Callback for new audio data
    ):
        """Initialize audio capture.
        
        Args:
            host: Source audio stream hostname
            port: Source audio stream port
            buffer_size: Socket buffer size
            original_rate: Original sample rate (Hz)
            target_rate: Target sample rate for downsampling (Hz)
            channels: Number of audio channels
            chunk_size: Audio chunk size in samples
            on_data: Callback function(original_chunk, downsampled_chunk) for new audio data
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.on_data = on_data
        
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        # Buffers for both streams
        self.original_buffer = []
        self.downsampled_buffer = []
        
        # Lock for thread-safe buffer access
        self.buffer_lock = threading.Lock()
        
        # Connection state
        self.connected = False
        self.last_data_time = 0
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 5.0
        self.data_timeout = 2.0  # Consider connection dead after 2 seconds of no data
        
        # Audio format info (float32)
        self.bytes_per_sample = channels * 4  # 4 bytes per float32
        self.samples_per_buffer = buffer_size // self.bytes_per_sample
        self.format_verified = False  # Track if we've verified the audio format
        
        # Initial buffer handling
        self.initial_buffer_processed = False
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
        
    def start(self) -> None:
        """Start audio capture."""
        if self.running:
            logger.warning("Audio capture already running")
            return
            
        try:
            self._connect()
            self.running = True
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            self._cleanup()
            raise
            
    def stop(self) -> None:
        """Stop audio capture gracefully."""
        logger.info("Stopping audio capture...")
        self.running = False
        self._cleanup()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)  # Wait up to 2 seconds
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
        logger.info("Audio capture stopped")
            
    def _connect(self) -> None:
        """Establish connection to audio source."""
        if self.socket:
            self._cleanup()
            
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5.0)  # 5 second timeout for operations
        
        try:
            # Resolve hostname first
            ip = socket.gethostbyname(self.host)
            logger.debug(f"Resolved {self.host} to {ip}")
            
            # Connect
            self.socket.connect((self.host, self.port))
            logger.info(f"Connected to audio stream at {self.host}:{self.port}")
            
            # Update connection state
            self.connected = True
            self.last_data_time = time.time()
            self.reconnect_delay = 1.0  # Reset delay after successful connection
            
        except Exception as e:
            self._cleanup()
            raise
            
    def _cleanup(self) -> None:
        """Clean up resources."""
        if self.socket:
            try:
                # Only try to shutdown if socket is still connected
                if self.connected:
                    try:
                        self.socket.shutdown(socket.SHUT_RDWR)
                    except OSError as e:
                        # Ignore connection reset and bad file descriptor errors during shutdown
                        if e.errno not in [errno.ECONNRESET, errno.EBADF]:
                            logger.warning(f"Error during socket shutdown: {e}")
            except Exception as e:
                logger.debug(f"Error during socket cleanup: {e}")
            finally:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.debug(f"Error closing socket: {e}")
                self.socket = None
        self.connected = False
            
    def _verify_audio_format(self, data: bytes) -> bool:
        """Verify the audio format from the first chunk of data.
        
        Args:
            data: Raw audio data bytes
            
        Returns:
            bool: True if format is valid
        """
        if self.format_verified:
            return True
            
        try:
            # Try to interpret as float32
            samples = np.frombuffer(data, dtype=np.float32)
            
            # Check if values are in reasonable range for float32 audio (-1 to 1)
            if samples.min() >= -1.1 and samples.max() <= 1.1:
                logger.info("Audio format verified as float32")
                self.format_verified = True
                return True
            else:
                logger.error(f"Invalid audio range: min={samples.min():.3f}, max={samples.max():.3f}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify audio format: {e}")
            return False
            
    def _capture_loop(self) -> None:
        """Main capture loop."""
        data_buffer = bytearray()  # Buffer for incomplete frames
        
        while self.running:
            try:
                if not self.socket:
                    self._connect()
                    
                # Check if we need to reconnect due to timeout
                if time.time() - self.last_data_time > self.data_timeout:
                    logger.warning("Data timeout, reconnecting...")
                    self._cleanup()
                    self._connect()
                    
                # Read raw audio data
                data = self.socket.recv(self.buffer_size)
                if not data:
                    logger.warning("No data received from stream")
                    raise ConnectionError("Empty data received")
                    
                # Update last data time
                self.last_data_time = time.time()
                
                # Verify format if not done yet
                if not self.format_verified and not self._verify_audio_format(data):
                    logger.error("Invalid audio format, reconnecting...")
                    self._cleanup()
                    continue
                
                # Add to buffer
                data_buffer.extend(data)
                
                # Process complete frames
                complete_frames = len(data_buffer) // self.bytes_per_sample
                if complete_frames > 0:
                    frame_bytes = complete_frames * self.bytes_per_sample
                    frame_data = data_buffer[:frame_bytes]
                    data_buffer = data_buffer[frame_bytes:]
                    
                    # Convert to numpy array (float32)
                    try:
                        audio_chunk = np.frombuffer(frame_data, dtype=np.float32)
                        if self.channels == 2:
                            # If stereo, take mean of channels
                            audio_chunk = audio_chunk.reshape(-1, 2).mean(axis=1)
                        
                        # Log first few samples for debugging
                        if len(audio_chunk) > 0:
                            logger.debug(f"First 5 samples: {audio_chunk[:5]}")
                            logger.debug(f"Min: {audio_chunk.min():.3f}, Max: {audio_chunk.max():.3f}, Mean: {audio_chunk.mean():.3f}")
                        
                        # Process both streams
                        self._process_audio_chunk(audio_chunk)
                        
                    except Exception as e:
                        logger.error(f"Error processing audio chunk: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                self._cleanup()
                
                if self.running:  # Only attempt reconnect if we're supposed to be running
                    # Exponential backoff for reconnection
                    time.sleep(self.reconnect_delay)
                    self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
                
    def _process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Process audio chunk for both streams.
        
        Args:
            chunk: Raw audio chunk as numpy array (float32)
        """
        with self.buffer_lock:
            # Store original (already float32)
            self.original_buffer.append(chunk.copy())
            
            # Downsample using librosa (already float32)
            downsampled = librosa.resample(
                y=chunk,
                orig_sr=self.original_rate,
                target_sr=self.target_rate
            )
            
            self.downsampled_buffer.append(downsampled)
            
            # Call data callback if set
            if self.on_data:
                try:
                    self.on_data(chunk, downsampled)
                except Exception as e:
                    logger.error(f"Error in data callback: {e}")
            
    def get_audio_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current audio data from both buffers.
        
        Returns:
            Tuple of (original_audio, downsampled_audio) as float32 arrays
        """
        with self.buffer_lock:
            if not self.original_buffer or not self.downsampled_buffer:
                return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
                
            original = np.concatenate(self.original_buffer)
            downsampled = np.concatenate(self.downsampled_buffer)
            
            # Initial buffer handling
            if not self.initial_buffer_processed:
                logger.info("Processing initial buffer of %d samples", len(original))
                self.initial_buffer_processed = True
                # Only keep the last second of data
                if len(original) > self.original_rate:
                    original = original[-self.original_rate:]
                if len(downsampled) > self.target_rate:
                    downsampled = downsampled[-self.target_rate:]
            
            # Clear buffers
            self.original_buffer.clear()
            self.downsampled_buffer.clear()
            
            return original, downsampled
