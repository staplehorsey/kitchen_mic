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

# Get module logger
logger = logging.getLogger(__name__)

def visualize_audio_level(audio_chunk: np.ndarray, width: int = 50) -> str:
    """Create a visual representation of audio level.
    
    Args:
        audio_chunk: Audio samples
        width: Width of visualization
        
    Returns:
        ASCII visualization of audio level
    """
    level = float(np.abs(audio_chunk).mean())
    normalized = min(1.0, level * 5)  # Amplify for better visibility
    bars = int(normalized * width)
    return f"[{'=' * bars}{' ' * (width - bars)}] ({level:.4f})"

class AudioCapture:
    """Captures audio from network stream and provides dual-stream processing."""
    
    def __init__(
        self,
        host: str = "staple.local",
        port: int = 12345,
        buffer_size: int = 4096,
        original_rate: int = 44000,  # Match server's sample rate
        target_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,  # Match server's chunk size
        visualize: bool = True
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
            visualize: Whether to show audio level visualization
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.visualize = visualize
        
        logger.info(f"AudioCapture initialized with visualize={visualize}")
        
        # Socket and thread state
        self.socket: Optional[socket.socket] = None
        self._running = False
        self.capture_thread: Optional[threading.Thread] = None
        self.connect_timeout = 5.0
        
        # Connection state
        self.connected = False
        self.last_data_time = 0
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 5.0
        self.data_timeout = 2.0  # Consider connection dead after 2 seconds of no data
        
        # Audio format info (float32)
        self.bytes_per_sample = channels * 4  # 4 bytes per float32
        self.samples_per_buffer = buffer_size // self.bytes_per_sample
        
        # Callbacks for audio data
        self._callbacks = []
        self._lock = threading.Lock()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
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
    
    def _connect(self) -> bool:
        """Connect to audio source."""
        try:
            logger.debug(f"Attempting to connect to {self.host}:{self.port}")
            
            # Create socket
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
                self.socket = None
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.connect_timeout)
            
            # Connect
            self.socket.connect((self.host, self.port))
            logger.debug("Successfully connected to audio source")
            
            # Set socket options
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # Set non-blocking after connect
            self.socket.setblocking(False)
            logger.debug("Set socket to non-blocking mode")
            
            self.connected = True
            self.last_data_time = time.time()
            self.reconnect_delay = 1.0  # Reset delay on successful connection
            return True
            
        except socket.error as e:
            logger.error(f"Failed to connect to audio source: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False

    def _process_audio(self, data: bytes) -> None:
        """Process raw audio data and notify callbacks."""
        try:
            logger.debug(f"Processing {len(data)} bytes of audio data")
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.float32)
            logger.debug(f"Converted to numpy array of shape {audio_data.shape}")
            
            # Resample for VAD if needed
            if self.target_rate != self.original_rate:
                logger.debug(f"Resampling from {self.original_rate}Hz to {self.target_rate}Hz")
                resampled = librosa.resample(
                    audio_data,
                    orig_sr=self.original_rate,
                    target_sr=self.target_rate
                )
            else:
                resampled = audio_data
                
            logger.debug(f"Resampled audio shape: {resampled.shape}")
            
            # Visualize audio level if enabled
            if self.visualize:
                viz = visualize_audio_level(audio_data)
                logger.debug(f"Audio level: {viz}")
            
            # Notify callbacks with timestamp and both audio streams
            timestamp = time.time()
            with self._lock:
                for callback in self._callbacks:
                    try:
                        logger.debug(f"Calling callback {callback.__name__}")
                        callback(timestamp, audio_data, resampled)
                    except Exception as e:
                        logger.error(f"Error in callback {callback.__name__}: {e}")
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}", exc_info=True)

    def _capture_loop(self) -> None:
        """Main audio capture loop."""
        logger.debug("Starting capture loop")
        iteration = 0
        
        while self._running:
            iteration += 1
            logger.debug(f"Capture loop iteration {iteration}")
            
            try:
                if not self.socket:
                    logger.debug("Socket not connected, attempting to connect...")
                    if not self._connect():
                        delay = self.reconnect_delay
                        logger.debug(f"Connection failed, waiting {delay:.1f}s before retry")
                        time.sleep(delay)
                        self.reconnect_delay = min(
                            self.reconnect_delay * 2,
                            self.max_reconnect_delay
                        )
                    continue
                
                # Check for data timeout
                now = time.time()
                time_since_data = now - self.last_data_time
                if time_since_data > self.data_timeout:
                    logger.warning(f"Data timeout after {time_since_data:.1f}s, reconnecting...")
                    self._cleanup()
                    continue
                
                # Read chunk size first (4 bytes)
                try:
                    logger.debug("Waiting for size header...")
                    size_data = self.socket.recv(4)
                    if not size_data:
                        logger.warning("Connection closed by server (no size data)")
                        self._cleanup()
                        continue
                    
                    chunk_size = struct.unpack('!I', size_data)[0]
                    logger.debug(f"Size header received: expecting {chunk_size} bytes")
                    
                    # Read audio data
                    data = b''
                    deadline = time.time() + 1.0  # 1 second timeout for complete chunk
                    read_attempts = 0
                    
                    while len(data) < chunk_size and time.time() < deadline:
                        read_attempts += 1
                        remaining = chunk_size - len(data)
                        try:
                            logger.debug(f"Attempt {read_attempts}: reading {remaining} remaining bytes")
                            chunk = self.socket.recv(min(remaining, self.buffer_size))
                            if not chunk:
                                logger.warning("Connection closed during data read")
                                raise ConnectionError("Connection closed during data read")
                            data += chunk
                            logger.debug(f"Read {len(chunk)} bytes, total {len(data)}/{chunk_size}")
                        except socket.error as e:
                            if e.errno == errno.EAGAIN:
                                # Non-blocking socket would block, try again
                                logger.debug("Socket would block, waiting...")
                                time.sleep(0.001)
                                continue
                            raise
                    
                    if len(data) == chunk_size:
                        logger.debug(f"Successfully read complete chunk of {len(data)} bytes")
                        self._process_audio(data)
                        self.last_data_time = time.time()
                    else:
                        logger.warning(f"Incomplete chunk after {read_attempts} attempts: got {len(data)}/{chunk_size} bytes")
                        raise ConnectionError("Incomplete chunk read")
                    
                except socket.error as e:
                    if e.errno == errno.EAGAIN:
                        logger.debug("Socket would block on size header, retrying...")
                        time.sleep(0.001)
                        continue
                    logger.error(f"Socket error: {e}", exc_info=True)
                    raise
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}", exc_info=True)
                self._cleanup()
                time.sleep(self.reconnect_delay)
                self.reconnect_delay = min(
                    self.reconnect_delay * 2,
                    self.max_reconnect_delay
                )

    def _cleanup(self) -> None:
        """Clean up socket connection."""
        logger.debug("Cleaning up socket connection")
        if self.socket:
            try:
                logger.debug("Closing socket")
                self.socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        self.socket = None
        self.connected = False
        logger.debug("Socket cleanup complete")
    
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
        self._running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return
        
        logger.info("Stopping audio capture")
        self._running = False
        
        # Clean up
        self._cleanup()
        
        # Wait for thread
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        
        logger.info("Audio capture stopped")
    
    @property
    def sample_rate(self) -> int:
        """Get the original sample rate."""
        return self.original_rate
