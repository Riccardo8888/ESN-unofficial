#!/usr/bin/env python3
"""
Real-time ESN Probability Visualizer
====================================

Displays the 37 angle bins with their probabilities in real-time
using PyGame bar chart visualization.

Reads prediction data from the running websocket_reservoir.py server.
"""

import pygame
import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from vnicktest.scripts.configuration import NUM_ANGLE_BINS, ANGLE_MIN, ANGLE_MAX, ANGLE_RESOLUTION

# Window dimensions
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 600
BAR_SPACING = 2
BAR_WIDTH = (WINDOW_WIDTH - (NUM_ANGLE_BINS + 1) * BAR_SPACING) // NUM_ANGLE_BINS

# Colors
BACKGROUND = (20, 20, 30)
BAR_COLOR = (100, 200, 255)
BAR_HIGHLIGHT = (255, 150, 50)  # For selected/sampled bin
GRID_COLOR = (50, 50, 60)
TEXT_COLOR = (200, 200, 200)
LABEL_COLOR = (150, 150, 150)

# Fonts
pygame.init()
FONT_LARGE = pygame.font.Font(None, 36)
FONT_MEDIUM = pygame.font.Font(None, 24)
FONT_SMALL = pygame.font.Font(None, 18)


class ProbabilityVisualizer:
    """Real-time visualization of ESN angle predictions."""
    
    def __init__(self):
        """Initialize PyGame window and visualization."""
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("ESN Angle Prediction - Real-time Probabilities")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Shared data from websocket_reservoir (will be imported)
        self.server = None
        self.latest_probs = np.ones(NUM_ANGLE_BINS) / NUM_ANGLE_BINS  # Uniform initially
        self.latest_boost = 0.0
        self.sampled_bin = None
        self.frame_count = 0
        
    def connect_to_server(self, server):
        """
        Connect to running websocket server to read predictions.
        
        Args:
            server: SlitherESNServer instance
        """
        self.server = server
        
    def update_data(self):
        """Pull latest prediction data from server."""
        if self.server is None:
            return
            
        # Get latest predictions
        if self.server.latest_angle_probs is not None:
            self.latest_probs = self.server.latest_angle_probs
            
        if self.server.latest_boost_prob is not None:
            self.latest_boost = self.server.latest_boost_prob
            
        if self.server.latest_command is not None:
            # Find which bin was sampled (from angle_delta)
            angle_delta = self.server.latest_command['angleDelta']
            angle_deg = np.degrees(angle_delta)
            angle_deg = np.clip(angle_deg, ANGLE_MIN, ANGLE_MAX)
            sampled_bin = int((angle_deg - ANGLE_MIN) / ANGLE_RESOLUTION)
            self.sampled_bin = np.clip(sampled_bin, 0, NUM_ANGLE_BINS - 1)
        
        if hasattr(self.server, 'stats'):
            self.frame_count = self.server.stats['frames_processed']
    
    def draw_bars(self):
        """Draw probability bars for each angle bin."""
        # Draw grid lines
        for i in range(0, WINDOW_HEIGHT, 50):
            pygame.draw.line(self.screen, GRID_COLOR, (0, i), (WINDOW_WIDTH, i), 1)
        
        # Calculate bar positions
        chart_height = WINDOW_HEIGHT - 150  # Leave space for labels
        chart_top = 80
        
        # Use FIXED scale: 0.4 probability = full chart height (no rescaling!)
        # This makes bars more visible since typical max is around 0.2-0.3
        max_prob = np.max(self.latest_probs)  # Still needed for text display
        SCALE_MAX = 0.4  # Full height at 40% probability
        
        # Draw each bar
        for i in range(NUM_ANGLE_BINS):
            # Calculate bar dimensions
            x = i * (BAR_WIDTH + BAR_SPACING) + BAR_SPACING
            prob = self.latest_probs[i]
            bar_height = int((prob / SCALE_MAX) * chart_height)  # FIXED: prob=0.4 → full height
            bar_height = min(bar_height, chart_height)  # Clip if > 0.4
            y = chart_top + chart_height - bar_height
            
            # Choose color (highlight sampled bin)
            color = BAR_HIGHLIGHT if i == self.sampled_bin else BAR_COLOR
            
            # Draw bar
            pygame.draw.rect(self.screen, color, (x, y, BAR_WIDTH, bar_height))
            
            # Draw probability text on top of high bars (use absolute threshold)
            if prob > 0.05:  # Show text if probability > 5%
                prob_text = FONT_SMALL.render(f"{prob:.3f}", True, TEXT_COLOR)
                text_rect = prob_text.get_rect(centerx=x + BAR_WIDTH // 2, bottom=y - 5)
                self.screen.blit(prob_text, text_rect)
        
        # Draw angle labels at bottom
        for i in range(0, NUM_ANGLE_BINS, 4):  # Every 4th bin
            angle = i * ANGLE_RESOLUTION + ANGLE_MIN
            x = i * (BAR_WIDTH + BAR_SPACING) + BAR_SPACING + BAR_WIDTH // 2
            label = FONT_SMALL.render(f"{int(angle)}°", True, LABEL_COLOR)
            label_rect = label.get_rect(centerx=x, top=chart_top + chart_height + 10)
            self.screen.blit(label, label_rect)
    
    def draw_info(self):
        """Draw header information."""
        # Title
        title = FONT_LARGE.render("ESN Angle Prediction Probabilities", True, TEXT_COLOR)
        self.screen.blit(title, (20, 20))
        
        # Frame count
        frame_text = FONT_MEDIUM.render(f"Frame: {self.frame_count}", True, TEXT_COLOR)
        self.screen.blit(frame_text, (WINDOW_WIDTH - 150, 20))
        
        # Boost probability
        boost_color = (255, 100, 100) if self.latest_boost > 0.5 else TEXT_COLOR
        boost_text = FONT_MEDIUM.render(f"Boost: {self.latest_boost:.2%}", True, boost_color)
        self.screen.blit(boost_text, (WINDOW_WIDTH - 180, 50))
        
        # Sampled angle
        if self.sampled_bin is not None:
            sampled_angle = self.sampled_bin * ANGLE_RESOLUTION + ANGLE_MIN
            angle_text = FONT_MEDIUM.render(f"Sampled: {sampled_angle:+.0f}°", True, BAR_HIGHLIGHT)
            self.screen.blit(angle_text, (20, 60))
        
        # Max probability
        max_prob = np.max(self.latest_probs)
        max_bin = np.argmax(self.latest_probs)
        max_angle = max_bin * ANGLE_RESOLUTION + ANGLE_MIN
        max_text = FONT_MEDIUM.render(f"Max: {max_prob:.3f} @ {max_angle:+.0f}°", True, TEXT_COLOR)
        self.screen.blit(max_text, (250, 60))
        
        # Footer
        footer = FONT_SMALL.render("Press ESC to exit", True, LABEL_COLOR)
        self.screen.blit(footer, (WINDOW_WIDTH // 2 - 60, WINDOW_HEIGHT - 30))
    
    def handle_events(self):
        """Handle PyGame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def render(self):
        """Render one frame."""
        # Clear screen
        self.screen.fill(BACKGROUND)
        
        # Draw visualization
        self.draw_bars()
        self.draw_info()
        
        # Update display
        pygame.display.flip()
    
    def run(self, update_rate=30):
        """
        Main visualization loop.
        
        Args:
            update_rate: Target FPS for visualization
        """
        print("\n" + "=" * 60)
        print("ESN PROBABILITY VISUALIZER")
        print("=" * 60)
        print(f"Window size: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        print(f"Angle bins: {NUM_ANGLE_BINS} ({ANGLE_MIN}° to {ANGLE_MAX}°)")
        print(f"Update rate: {update_rate} FPS")
        print("\nPress ESC to exit")
        print("=" * 60)
        
        while self.running:
            # Handle events
            self.handle_events()
            
            # Update data from server
            self.update_data()
            
            # Render frame
            self.render()
            
            # Maintain frame rate
            self.clock.tick(update_rate)
        
        pygame.quit()
        print("\n✓ Visualizer closed")


def main_standalone():
    """Run visualizer in standalone mode (simulated data)."""
    print("Running in STANDALONE mode (simulated data)")
    visualizer = ProbabilityVisualizer()
    
    # Simulate data updates
    def simulate_data():
        # Create peaked distribution
        center = np.random.randint(0, NUM_ANGLE_BINS)
        probs = np.exp(-0.5 * ((np.arange(NUM_ANGLE_BINS) - center) / 3) ** 2)
        probs /= np.sum(probs)
        visualizer.latest_probs = probs
        visualizer.latest_boost = np.random.rand()
        visualizer.sampled_bin = np.random.choice(NUM_ANGLE_BINS, p=probs)
        visualizer.frame_count += 1
    
    # Override update_data to use simulation
    original_update = visualizer.update_data
    visualizer.update_data = lambda: (original_update(), simulate_data())[1]
    
    visualizer.run(update_rate=10)  # Slower for simulation


if __name__ == '__main__':
    # Run in standalone mode if executed directly
    main_standalone()
