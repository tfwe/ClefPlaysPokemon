import sys
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig

class ClefAgent:
    def __init__(self, model_id: str = "Cloudflare/clef-flash"):
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

    def decide(self, frame: Image.Image, context: str) -> tuple[dict, str, str, dict]:
        scaled_frame = frame.resize((480, 320), Image.Resampling.NEAREST)

        payload = {
            "model": "clef-flash",
            "state": context,
            "images": [scaled_frame],
            "questions": {
                "scene_type": {
                    "type": "choice",
                    "instructions": "Identify the primary screen state currently visible in the game.",
                    "criteria": {
                        "TITLE": "Title screen, logo, or opening cutscene.",
                        "DIALOG": "Dialogue text box with an arrow or NPC talking at the bottom.",
                        "MENU": "Start menu, item bag, Pokemon list, or selection window.",
                        "BATTLE": "Turn-based combat screen showing Pokemon sprites and HP bars.",
                        "OVERWORLD": "Character walking in rooms, moving truck, towns, or routes."
                    }
                },
                "next_button": {
                    "type": "choice",
                    "instructions": (
                        "Select the single best button to make game progress. "
                        "If stuck in an enclosed space or truck, look for the doorway/exit and move toward it. "
                        "Do not bump into walls or boxes."
                    ),
                    "criteria": {
                        "A": "Confirm, advance dialog, interact with NPCs/objects.",
                        "B": "Cancel menu or hold to run.",
                        "UP": "Walk up toward doorways, stairs, or northern exits.",
                        "DOWN": "Walk down toward exits or southern paths.",
                        "LEFT": "Walk left toward exits or western paths.",
                        "RIGHT": "Walk right toward exits or eastern paths.",
                        "START": "Open the in-game pause menu.",
                        "SELECT": "Use registered key item."
                    }
                }
            }
        }

        with torch.inference_mode():
            response = self._systemone(self.model, self.processor, payload)

        answers = response.get("answers", {})

        # Scene probs
        scene_data = answers.get("scene_type", {})
        scene_probs = scene_data.get("probabilities") or scene_data.get("probs") or {k: v for k, v in scene_data.items() if isinstance(v, (int, float))}
        detected_scene = max(scene_probs, key=scene_probs.get) if scene_probs else "OVERWORLD"

        # Action probs
        btn_data = answers.get("next_button", {})
        btn_probs = btn_data.get("probabilities") or btn_data.get("probs") or {k: v for k, v in btn_data.items() if isinstance(v, (int, float))}

        if detected_scene == "TITLE":
            for direction in ["UP", "DOWN", "LEFT", "RIGHT"]:
                if direction in btn_probs:
                    btn_probs[direction] *= 0.05

        chosen_button = max(btn_probs, key=btn_probs.get)
        return scene_probs, detected_scene, chosen_button, btn_probs
