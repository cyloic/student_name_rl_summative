# environment/rendering.py

import pygame
import numpy as np

class SmartSortRenderer:
    def __init__(self, screen_width=600, screen_height=300):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("SmartSort RL Agent Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 30)

    def render_frame(self, state, true_class, current_step):
        # State format: [Texture, Shape, Conf_Plastic, Conf_Paper]
        
        # --- Colors ---
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GREY = (200, 200, 200)
        GREEN = (0, 150, 0)
        RED = (150, 0, 0)

        self.screen.fill(WHITE)

        # --- Display Meta Information ---
        meta_text = f"Step: {current_step} | True Class: {'Plastic' if true_class == 0 else 'Paper'}"
        text_surface = self.font.render(meta_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        # --- Bar Chart Visualization ---
        bar_names = ["Texture Feature", "Shape Feature", "PLASTIC Confidence", "PAPER Confidence"]
        scores = state 
        num_scores = len(scores)
        
        bar_width = (self.screen_width - 80) // num_scores
        bar_height_max = self.screen_height - 100
        
        for i in range(num_scores):
            # Calculate bar dimensions
            bar_x = 40 + i * (bar_width + 10)
            bar_height = int(scores[i] * bar_height_max)
            bar_y = self.screen_height - 40 - bar_height
            
            # Choose color
            if i >= 2:
                # Confidence bars (2=Plastic, 3=Paper)
                color = GREEN if i == (2 + true_class) else RED
            else:
                # Feature bars
                color = GREY
                
            # Draw the bar
            pygame.draw.rect(self.screen, color, (bar_x, bar_y, bar_width, bar_height))
            
            # Draw name label
            label = self.font.render(bar_names[i], True, BLACK)
            self.screen.blit(label, (bar_x, self.screen_height - 35))
            
            # Draw score label
            score_label = self.font.render(f"{scores[i]:.2f}", True, BLACK)
            self.screen.blit(score_label, (bar_x + bar_width/2 - score_label.get_width()/2, bar_y - 25))

        pygame.display.flip()
        self.clock.tick(60) 

    def close(self):
        pygame.quit()