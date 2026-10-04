import sys
import torch
import random
from collections import deque
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig

class ClefAgent:
    def __init__(self, model_id: str = "Cloudflare/clef-flash", history_len: int = 10):
        print(f"Loading {model_id} in 4-bit NF4...")
        self.model_path = snapshot_download(model_id)
        sys.path.insert(0, self.model_path)
        
        from joint_schema_model import load_release_model, systemone
        self._systemone = systemone

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["visual", "lm_head"]
        )

        self.model, self.processor = load_release_model(
            self.model_path,
            device="cuda:0",
            quantization_config=bnb_config
        )
        print("Clef-Flash 4-bit loaded successfully.")

        # --- NEW: Memory Systems ---
        self.action_history = deque(maxlen=history_len)
        self.scene_history = deque(maxlen=history_len)
        self.current_objective = "Explore the current area, find the exit, or talk to people."
        self.temperature = 0.8 # Controls randomness: higher = more random, lower = strictly follows VLM

    def set_objective(self, new_objective: str):
        """Allow an external script to update the long-term goal."""
        self.current_objective = new_objective

    def decide(self, frame: Image.Image, external_context: str = "") -> tuple[dict, str, str, dict]:
        scaled_frame = frame.resize((480, 320), Image.Resampling.NEAREST)

        # Build a much richer state description for the VLM
        recent_actions = ", ".join(list(self.action_history)[-5:]) if self.action_history else "None"
        rich_context = (
            f"Objective: {self.current_objective}. "
            f"Recent actions taken: {recent_actions}. "
            f"Additional context: {external_context}"
        )

        payload = {
            "model": "clef-flash",
            "state": rich_context,
            "images": [scaled_frame],
            "questions": {
                "scene_type": {
                    "type": "choice",
                    "instructions": "Identify the primary screen state currently visible in the game.",
                    "criteria": {
                        "TITLE": "Title screen, logo, or opening cutscene.",
                        "DIALOG": "Dialogue text box at the bottom of the screen.",
                        "NAMING_SCREEN": "A keyboard layout on screen asking to input a name.",
                        "MENU_MAIN": "The main START menu (Pokedex, Pokemon, Bag, Save).",
                        "MENU_BAG": "Inside the item bag looking at items.",
                        "BATTLE": "Turn-based combat screen showing Pokemon sprites and HP bars.",
                        "OVERWORLD": "Character walking in rooms, moving truck, towns, or routes."
                    }
                },
                "next_button": {
                    "type": "choice",
                    "instructions": (
                        "Select the single best button to make progress towards the objective. "
                        "Do NOT repeat recent actions if you are stuck in a corner."
                    ),
                    "criteria": {
                        "A": "Confirm, advance dialog, interact with NPCs/objects.",
                        "B": "Cancel menu, back out, or hold to run.",
                        "UP": "Walk up.",
                        "DOWN": "Walk down.",
                        "LEFT": "Walk left.",
                        "RIGHT": "Walk right.",
                        "START": "Open the in-game pause menu or confirm name entries.",
                        "SELECT": "Use registered key item."
                    }
                }
            }
        }

        with torch.inference_mode():
            response = self._systemone(self.model, self.processor, payload)

        answers = response.get("answers", {})

        # Parse Scene
        scene_data = answers.get("scene_type", {})
        scene_probs = scene_data.get("probabilities") or scene_data.get("probs") or {k: v for k, v in scene_data.items() if isinstance(v, (int, float))}
        detected_scene = max(scene_probs, key=scene_probs.get) if scene_probs else "OVERWORLD"

        # Parse Actions
        btn_data = answers.get("next_button", {})
        btn_probs = btn_data.get("probabilities") or btn_data.get("probs") or {k: v for k, v in btn_data.items() if isinstance(v, (int, float))}

        # --- NEW: Probabilistic Sampling & Emergent Macros ---
        
        # 1. History Decay (Universal Loop Penalty)
        # Instead of hardcoding scenes, we penalize repeating the same actions over and over.
        # This naturally creates sequences (e.g., if START is pressed, its prob drops next frame, allowing A to be pressed)
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            if last_action in btn_probs:
                # Immediate decay: discourage holding the same button back-to-back unconditionally
                btn_probs[last_action] *= 0.4 
            
            # Rut penalty: If we've mashed the exact same button 3+ times in recent memory, heavily penalize it
            last_5 = list(self.action_history)[-5:]
            for btn in btn_probs.keys():
                if last_5.count(btn) >= 3:
                    btn_probs[btn] *= 0.05

        # 2. Temperature Sampling
        # Select action based on the distribution rather than deterministically picking the absolute highest
        actions = list(btn_probs.keys())
        weights = list(btn_probs.values())
        
        if sum(weights) > 0:
            # Apply temperature and a small epsilon (0.02) to ensure no probability is strictly 0%
            # This allows for "breakout" random actions if the agent gets truly stuck
            epsilon = 0.02
            adjusted_weights = [(w ** (1.0 / self.temperature)) + epsilon for w in weights]
            chosen_button = random.choices(actions, weights=adjusted_weights, k=1)[0]
        else:
            # Fallback if the VLM returns an empty or invalid dictionary
            chosen_button = random.choice(["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START"])

        # Update History
        self.action_history.append(chosen_button)
        self.scene_history.append(detected_scene)

        return scene_probs, detected_scene, chosen_button, btn_probs
