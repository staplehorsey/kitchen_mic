"""Base visualization components for Kitchen Mic."""

import logging
import threading
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class BaseVisualizer(ABC):
    """Base class for real-time visualizations.
    
    Provides common functionality for:
    - Dark theme setup
    - Figure and subplot management
    - Standard controls (Q to quit, M to mute)
    - Animation handling
    - Thread-safe data management
    """
    
    def __init__(
        self,
        num_subplots: int = 1,
        figsize: Tuple[int, int] = (12, 8),
        window_size: int = 2000,
        title: str = "Real-time Visualization"
    ):
        """Initialize base visualizer.
        
        Args:
            num_subplots: Number of subplot rows
            figsize: Figure size (width, height) in inches
            window_size: Number of data points to display
            title: Figure title
        """
        self.window_size = window_size
        self.running = True
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=figsize)
        if num_subplots == 1:
            self.axes = [self.axes]  # Make it a list for consistent access
        self.fig.suptitle(title, fontsize=14)
        
        # Status text
        self._add_status_text()
        
        # Audio player reference for mute toggle
        self.audio_player = None
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation
        self.anim: Optional[FuncAnimation] = None
        
        # Window events
        self._setup_events()
    
    def _add_status_text(self) -> None:
        """Add status text to figure."""
        self.status_text = self.fig.text(
            0.02, 0.02,
            'Press Q to quit\nPress M to toggle audio',
            color='white',
            alpha=0.7
        )
    
    def _setup_events(self) -> None:
        """Set up window event handlers."""
        self._close_cid = self.fig.canvas.mpl_connect('close_event', self._on_close)
        self._key_cid = self.fig.canvas.mpl_connect('key_press_event', self._on_key)
    
    def _on_key(self, event) -> None:
        """Handle key press events.
        
        Args:
            event: Matplotlib key event
        """
        if event.key == 'q':
            self.running = False
            plt.close(self.fig)
        elif event.key == 'm' and self.audio_player:
            self.audio_player.toggle_mute()
    
    def _on_close(self, event) -> None:
        """Handle window close event.
        
        Args:
            event: Matplotlib close event
        """
        self.running = False
    
    @abstractmethod
    def _update(self, frame) -> tuple:
        """Update animation frame.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple of artists that were modified
        """
        pass
    
    def start_animation(self) -> None:
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=50,  # Update every 50ms
            blit=False,
            cache_frame_data=False
        )
    
    def set_audio_player(self, player) -> None:
        """Set audio player reference for mute toggle.
        
        Args:
            player: Audio player instance
        """
        self.audio_player = player
    
    def show(self) -> None:
        """Show visualization window."""
        plt.show()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.anim:
            self.anim.event_source.stop()
        plt.close(self.fig)
    
    def configure_axis(
        self,
        ax: Axes,
        title: str,
        ylabel: str,
        ylim: Tuple[float, float] = (-1, 1),
        grid_alpha: float = 0.3
    ) -> None:
        """Configure a subplot axis with common settings.
        
        Args:
            ax: Matplotlib axis to configure
            title: Subplot title
            ylabel: Y-axis label
            ylim: Y-axis limits (min, max)
            grid_alpha: Grid line transparency
        """
        ax.set_title(title)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=grid_alpha)
        ax.legend(loc='upper right')
