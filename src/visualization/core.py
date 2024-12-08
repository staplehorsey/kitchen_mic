"""Core visualization components for Kitchen Mic."""

import logging
import threading
from typing import Optional, Tuple, List, Protocol, Callable
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

class PlotComponent(Protocol):
    """Protocol for plot components that can be added to a figure."""
    
    def setup(self, fig: Figure, ax: Axes) -> None:
        """Set up the plot component.
        
        Args:
            fig: Matplotlib figure
            ax: Matplotlib axes to plot on
        """
        ...
    
    def update(self, frame: int) -> tuple:
        """Update the plot for animation.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple of artists that were modified
        """
        ...
    
    def cleanup(self) -> None:
        """Clean up resources."""
        ...

class Visualizer:
    """Core visualization manager.
    
    Features:
    - Dark theme setup
    - Figure and subplot management
    - Standard controls (Q to quit, M to mute)
    - Animation handling
    - Thread-safe data management
    """
    
    def __init__(
        self,
        components: List[PlotComponent],
        num_subplots: int = 1,
        figsize: Tuple[int, int] = (12, 8),
        title: str = "Real-time Visualization"
    ):
        """Initialize visualizer.
        
        Args:
            components: List of plot components to manage
            num_subplots: Number of subplot rows
            figsize: Figure size (width, height) in inches
            title: Figure title
        """
        self.components = components
        self.running = True
        
        # Set up plot
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(num_subplots, 1, figsize=figsize)
        if num_subplots == 1:
            self.axes = [self.axes]  # Make it a list for consistent access
        self.fig.suptitle(title, fontsize=14)
        
        # Status text
        self.status_text = self.fig.text(
            0.02, 0.02,
            'Press Q to quit\nPress M to toggle audio',
            color='white',
            alpha=0.7
        )
        
        # Audio player reference for mute toggle
        self.audio_player = None
        
        # Lock for thread safety
        self.data_lock = threading.Lock()
        
        # Animation
        self.anim: Optional[FuncAnimation] = None
        
        # Set up components
        for component, ax in zip(components, self.axes):
            component.setup(self.fig, ax)
        
        # Window events
        self._setup_events()
    
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
    
    def _update(self, frame: int) -> tuple:
        """Update animation frame.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple of artists that were modified
        """
        artists = []
        for component in self.components:
            artists.extend(component.update(frame))
        return tuple(artists)
    
    def start_animation(self) -> None:
        """Start the animation."""
        self.anim = FuncAnimation(
            self.fig,
            self._update,
            interval=50,  # Update every 50ms
            blit=True,
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
        for component in self.components:
            component.cleanup()
        plt.close(self.fig)
    
    @staticmethod
    def configure_axis(
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
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=grid_alpha)
        ax.set_facecolor('black')
