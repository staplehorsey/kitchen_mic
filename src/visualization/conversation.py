"""Visualization module for conversation detection."""

import logging
import collections
import time
from typing import List, Optional, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .base import BaseVisualizer

logger = logging.getLogger(__name__)

class ConversationVisualizer(BaseVisualizer):
    """Visualizer for conversations with VAD data."""
    
    def __init__(self, window_size: int, samples_per_second: int, time_window: float, title: str = "Conversation Visualization"):
        """Initialize the conversation visualizer.
        
        Args:
            window_size: Number of samples to show in the window
            samples_per_second: Number of samples per second for VAD data
            time_window: Time window to display in seconds
            title: Title for the visualization window
        """
        # Initialize parent with proper parameters
        super().__init__(
            num_subplots=2,  # Audio + VAD
            figsize=(12, 8),
            window_size=window_size,
            title=title
        )
        
        # Store parameters
        self.samples_per_second = samples_per_second
        self.time_window = time_window
        
        # Initialize data buffers
        self.audio_data = collections.deque(maxlen=window_size)
        self.vad_data = collections.deque(maxlen=window_size)
        self.timestamps = collections.deque(maxlen=window_size)
        
        # Initialize with zeros
        current_time = time.time()
        for _ in range(window_size):
            self.audio_data.append(0.0)
            self.vad_data.append((current_time, 0.0))
            self.timestamps.append(current_time)
        
        # Conversation tracking
        self.conversations: List[Dict] = []
        self.selected_conversation: Optional[int] = None
        self.audio_player = None
        
        # Create subplot for conversation list
        self.fig.subplots_adjust(right=0.8)
        self.text_ax = self.fig.add_axes([0.82, 0.1, 0.15, 0.8])
        self.text_ax.axis('off')
        
        # Add key bindings
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        # Configure axes
        self.configure_axis(
            self.axes[0],
            title="Audio Waveform",
            ylabel="Amplitude",
            ylim=(-1, 1)
        )
        self.configure_axis(
            self.axes[1],
            title="Voice Activity",
            ylabel="VAD State",
            ylim=(-0.1, 1.1)
        )
        
        # Create line objects
        self.line_audio, = self.axes[0].plot([], [], 'cyan', label='Audio', linewidth=1)
        self.line_vad, = self.axes[1].plot([], [], 'lime', label='VAD State', linewidth=2)
    
    def run(self):
        """Run the visualization."""
        self.start_animation()
        self.show()
    
    def add_conversation(self, conversation: Dict):
        """Add a new conversation to track.
        
        Args:
            conversation: Dictionary containing conversation data
        """
        self.conversations.append(conversation)
        self._update_conversation_list()
    
    def _update_conversation_list(self):
        """Update the conversation list display."""
        self.text_ax.clear()
        self.text_ax.axis('off')
        
        text = "Conversations:\n\n"
        for i, conv in enumerate(self.conversations):
            prefix = "→ " if i == self.selected_conversation else "  "
            duration = conv['end_time'] - conv['start_time']
            text += f"{prefix}Conv {i}: {duration:.1f}s\n"
        
        self.text_ax.text(0, 1, text, va='top', fontfamily='monospace')
        self.fig.canvas.draw_idle()
    
    def _on_key(self, event):
        """Handle key press events.
        
        Args:
            event: Matplotlib key event
        """
        if event.key == 'up':
            if self.conversations:
                if self.selected_conversation is None:
                    self.selected_conversation = len(self.conversations) - 1
                else:
                    self.selected_conversation = (self.selected_conversation - 1) % len(self.conversations)
                self._update_conversation_list()
                
        elif event.key == 'down':
            if self.conversations:
                if self.selected_conversation is None:
                    self.selected_conversation = 0
                else:
                    self.selected_conversation = (self.selected_conversation + 1) % len(self.conversations)
                self._update_conversation_list()
                
        elif event.key in ['enter', ' ']:
            if self.selected_conversation is not None and self.audio_player:
                conv = self.conversations[self.selected_conversation]
                self.audio_player.play(conv['audio'])
                
        else:
            # Pass other keys to parent
            super()._on_key(event)
    
    def _draw_frame(self):
        """Draw the current frame."""
        super()._draw_frame()
        
        # Draw conversation markers
        with self.data_lock:
            current_time = time.time()
            window_start = current_time - self.time_window
            
            for conv in self.conversations:
                # Convert to window-relative coordinates
                start_x = max(0, conv['start_time'] - window_start)
                end_x = min(self.time_window, conv['end_time'] - window_start)
                
                if 0 <= end_x and start_x <= self.time_window:
                    rect = Rectangle(
                        (start_x, -0.1),
                        end_x - start_x,
                        1.2,
                        alpha=0.2,
                        color='green'
                    )
                    self.axes[1].add_patch(rect)
    
    def _update(self, frame) -> tuple:
        """Update animation frame.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple of artists that were modified
        """
        with self.data_lock:
            # Calculate current time window
            current_time = time.time()
            window_start = current_time - self.time_window
            window_end = current_time
            
            # Filter VAD data to current time window
            if len(self.vad_data) > 0:
                # Get all data points
                vad_times, vad_states = zip(*self.vad_data)
                
                # Convert to numpy arrays for easier filtering
                vad_times = np.array(vad_times)
                vad_states = np.array(vad_states)
                
                # Filter to current window
                vad_mask = (vad_times >= window_start) & (vad_times <= window_end)
                
                # Update VAD plot
                self.line_vad.set_data(
                    vad_times[vad_mask] - window_start,  # Normalize to [0, time_window]
                    vad_states[vad_mask]
                )
            
            # Update audio waveform
            times = np.linspace(0, self.time_window, len(self.audio_data))
            self.line_audio.set_data(times, list(self.audio_data))
            
            # Update axis limits to show scrolling window
            self.axes[0].set_xlim(0, self.time_window)
            self.axes[1].set_xlim(0, self.time_window)
        
        return self.line_audio, self.line_vad
    
    def add_data(self, timestamp: float, original: np.ndarray, vad_state: bool) -> None:
        """Add new data point.
        
        Args:
            timestamp: Current timestamp
            original: Original audio data
            vad_state: Current VAD state
        """
        with self.data_lock:
            # Add audio samples
            for sample in original:
                self.audio_data.append(sample)
            
            # Add VAD data with current timestamp
            current_time = time.time()
            self.vad_data.append((current_time, float(vad_state)))
