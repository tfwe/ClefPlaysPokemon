import cv2
import numpy as np
from PIL import Image

class VisualDashboard:
    def __init__(self, width: int = 1240, height: int = 540):
        self.width = width
        self.height = height

    def _draw_bar_group(self, canvas, title, items, selected_key, start_x, start_y, bar_max_w=200, row_gap=28):
        cv2.putText(canvas, title, (start_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

        sorted_items = sorted(items.items(), key=lambda x: x[1], reverse=True)
        for idx, (label, prob) in enumerate(sorted_items):
            y = start_y + 18 + (idx * row_gap)
            is_active = (label == selected_key)
            bar_color = (0, 215, 255) if is_active else (75, 75, 90)
            text_color = (255, 255, 255) if is_active else (160, 160, 160)

            # Label text
            cv2.putText(canvas, f"{label:<9}", (start_x, y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1 if not is_active else 2, cv2.LINE_AA)

            # Empty bar background
            bar_left = start_x + 95
            cv2.rectangle(canvas, (bar_left, y), (bar_left + bar_max_w, y + 16), (35, 35, 40), -1)

            # Filled bar
            filled_w = int(bar_max_w * max(0.0, min(1.0, prob)))
            if filled_w > 0:
                cv2.rectangle(canvas, (bar_left, y), (bar_left + filled_w, y + 16), bar_color, -1)

            # Percentage text (guaranteed margin)
            cv2.putText(canvas, f"{prob * 100:5.1f}%", (bar_left + bar_max_w + 12, y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA)

    def render(self, pil_frame: Image.Image, scene_probs: dict, detected_scene: str,
               chosen_action: str, action_probs: dict, latency_ms: float, step: int, coords: tuple[int, int]) -> np.ndarray:
        
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 22)

        # 1. Scaled Game Screen (240x160 -> 480x320)
        game_rgb = np.array(pil_frame)
        game_bgr = cv2.cvtColor(game_rgb, cv2.COLOR_RGB2BGR)
        game_resized = cv2.resize(game_bgr, (480, 320), interpolation=cv2.INTER_NEAREST)

        canvas[80:400, 25:505] = game_resized
        cv2.rectangle(canvas, (23, 78), (507, 402), (65, 65, 75), 2)

        # 2. Header & Telemetry
        cv2.putText(canvas, "CLEF PLAYS POKEMON EMERALD", (25, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.putText(canvas, f"Mode: {detected_scene} | Pos: ({coords[0]}, {coords[1]}) | Step: {step:05d} | Latency: {latency_ms:.1f}ms", 
                    (25, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1, cv2.LINE_AA)

        # 3. State Distribution (Column 1)
        self._draw_bar_group(canvas, "STATE CLASSIFICATION", scene_probs, detected_scene,
                             start_x=540, start_y=75, bar_max_w=170, row_gap=30)

        # 4. Action Distribution (Column 2)
        self._draw_bar_group(canvas, "ACTION DISTRIBUTION", action_probs, chosen_action,
                             start_x=890, start_y=75, bar_max_w=170, row_gap=30)

        return canvas
