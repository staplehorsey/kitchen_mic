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
        visualize: bool = True,
        visualization_width: int = 50,
        visualization_interval: float = 0.1
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
            visualization_width: Width of visualization
            visualization_interval: Update interval for visualization
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.visualize = visualize
        self.visualization_width = visualization_width
        self.visualization_interval = visualization_interval
        
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
        # Ensure buffer size is multiple of bytes_per_sample
        self.buffer_size = (buffer_size // self.bytes_per_sample) * self.bytes_per_sample
        self.samples_per_buffer = self.buffer_size // self.bytes_per_sample
        
        # Ensure chunk size is multiple of bytes_per_sample
        # chunk_size is in samples, convert to bytes
        self.samples_per_chunk = chunk_size
        self.chunk_size = self.samples_per_chunk * self.bytes_per_sample
        
        # Audio level visualization
        self.last_level_time = 0
        self.level_interval = self.visualization_interval  # Update level more frequently for smooth display
        self.last_level = ""  # Store last level for cleanup
        
        logger.info(
            f"Audio format: {channels} channels, {original_rate}Hz, "
            f"chunk_size={self.chunk_size} bytes ({self.samples_per_chunk} samples)"
        )
        
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
        """Process raw audio data and notify callbacks.
        
        Args:
            data: Raw audio bytes
        """
        try:
            # Convert to float32 array
            audio = np.frombuffer(data, dtype=np.float32)
            
            # Current wall clock time
            current_time = time.time()
            
            # Downsample for VAD
            downsampled = librosa.resample(
                audio,
                orig_sr=self.original_rate,
                target_sr=self.target_rate
            )
            
            # Visualize audio level if enabled
            if self.visualize:
                now = time.time()
                if now - self.last_level_time >= self.level_interval:
                    viz = visualize_audio_level(audio, width=self.visualization_width)
                    # Clear previous line and print new level
                    print(f"\r{' ' * len(self.last_level)}\r{viz}", end="", flush=True)
                    self.last_level = viz
                    self.last_level_time = now
            
            # Notify callbacks
            with self._lock:
                for callback in self._callbacks:
                    try:
                        callback(current_time, audio, downsampled)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
            
            self.last_data_time = current_time
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def _capture_loop(self) -> None:
        """Main audio capture loop."""
        logger.debug("Starting capture loop")
        
        while self._running:
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
                
                # Get expected chunk size from socket
                chunk_size = self.chunk_size
                if chunk_size > 10 * 1024 * 1024:  # > 10MB
                    logger.error(f"Invalid chunk size: {chunk_size/1024/1024:.1f}MB")
                    raise ValueError("Chunk size too large")
                
                # Read audio data
                data = b''
                deadline = time.time() + 1.0  # 1 second timeout for complete chunk
                read_start = time.time()
                total_reads = 0
                total_blocks = 0
                
                while len(data) < chunk_size and time.time() < deadline:
                    remaining = chunk_size - len(data)
                    try:
                        chunk = self.socket.recv(min(remaining, self.buffer_size))
                        if not chunk:
                            logger.warning("Connection closed during data read")
                            raise ConnectionError("Connection closed during data read")
                        data += chunk
                        total_reads += 1
                    except socket.error as e:
                        if e.errno == errno.EAGAIN:
                            total_blocks += 1
                            time.sleep(0.001)
                            continue
                        raise
                
                read_time = time.time() - read_start
                if len(data) == chunk_size:
                    # Validate data size is multiple of sample size
                    if len(data) % self.bytes_per_sample != 0:
                        logger.error(f"Data size {len(data)} not multiple of sample size {self.bytes_per_sample}")
                        raise ValueError("Invalid data size")
                        
                    # Only log details periodically to avoid spam
                    if total_reads > 100 or read_time > 0.1:
                        logger.debug(
                            f"Read {len(data)/1024:.1f}KB in {read_time*1000:.0f}ms "
                            f"({total_reads} reads, {total_blocks} blocks)"
                        )
                    self._process_audio(data)
                    self.last_data_time = time.time()
                else:
                    logger.warning(
                        f"Incomplete chunk: {len(data)/1024:.1f}KB/{chunk_size/1024:.1f}KB "
                        f"after {read_time*1000:.0f}ms"
                    )
                    raise ConnectionError("Incomplete chunk read")
                    
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
        self._running = True
        
        # Start capture thread
        logger.debug("Starting capture thread")
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True  # Make thread daemon so it exits with main program
        self.capture_thread.start()
        logger.debug("Capture thread started")
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self._running:
            return
        
        logger.info("Stopping audio capture")
        self._running = False
        
        if self.visualize:
            # Clear the audio level line when stopping
            print(f"\r{' ' * len(self.last_level)}\r", end="", flush=True)
        
        self._cleanup()
        
        # Wait for thread
        if self.capture_thread:
            logger.debug("Waiting for capture thread to stop")
            self.capture_thread.join(timeout=5.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
            self.capture_thread = None
        
        logger.info("Audio capture stopped")
    
    @property
    def sample_rate(self) -> int:
        """Get the original sample rate."""
        return self.original_rate
