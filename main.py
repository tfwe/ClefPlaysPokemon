import time
import threading
import random
import cv2
from emulator.engine import GBAEngine, BUTTON_IDS
from agent.clef_runner import ClefAgent
from broadcast.overlay import VisualDashboard

ROM_PATH = "roms/pokemon_emerald.gba"

state_lock = threading.Lock()
latest_scene = "INITIALIZING"
latest_scene_probs = {}
latest_action = "A"
latest_probs = {}
latest_latency = 0.0
latest_coords = (0, 0)
step_counter = 0
running = True

def inference_worker(engine: GBAEngine, agent: ClefAgent):
    global latest_scene, latest_scene_probs, latest_action, latest_probs, latest_latency, latest_coords, step_counter, running

    pos_history = []
    stuck_counter = 0

    while running:
        t0 = time.perf_counter()
        
        with state_lock:
            frame = engine.get_frame()
            context = engine.get_ram_context()
            coords = engine.get_player_coords()

        # Track player movement to detect wall bumps
        pos_history.append(coords)
        if len(pos_history) > 5:
            pos_history.pop(0)

        # If coords stayed identical across 4 consecutive steps in overworld
        is_stuck = len(pos_history) >= 4 and len(set(pos_history)) == 1 and coords != (0, 0)

        augmented_context = context
        if is_stuck:
            stuck_counter += 1
            augmented_context += f" | WARNING: Player is STUCK at {coords} bumping into an obstacle. Choose a different direction to get unstuck."
        else:
            stuck_counter = 0

        scene_probs, scene, action, probs = agent.decide(frame, augmented_context)
        latency = (time.perf_counter() - t0) * 1000

        # Hard override if stuck for too long
        if is_stuck and stuck_counter >= 3:
            # Force perpendicular escape move
            opposites = {"LEFT": "RIGHT", "RIGHT": "LEFT", "UP": "DOWN", "DOWN": "UP"}
            escape_options = ["DOWN", "RIGHT"] if action in ["UP", "LEFT"] else ["UP", "LEFT"]
            action = random.choice(escape_options)
            stuck_counter = 0

        with state_lock:
            latest_scene = scene
            latest_scene_probs = scene_probs
            latest_action = action
            latest_probs = probs
            latest_latency = latency
            latest_coords = coords
            step_counter += 1

def main():
    global running
    engine = GBAEngine(ROM_PATH)
    agent = ClefAgent()
    dashboard = VisualDashboard(width=1240, height=540)

    cv2.namedWindow("ClefPlaysPokemon", cv2.WINDOW_AUTOSIZE)
    worker = threading.Thread(target=inference_worker, args=(engine, agent), daemon=True)
    worker.start()

    print("Emerald loop active. Press 'q' in window to quit.")
    frame_interval = 1.0 / 60.0
    action_ticks = 0

    try:
        while True:
            t_frame_start = time.perf_counter()

            with state_lock:
                current_scene = latest_scene
                current_scene_probs = latest_scene_probs.copy()
                current_action = latest_action
                current_probs = latest_probs.copy()
                current_latency = latest_latency
                current_coords = latest_coords
                current_step = step_counter

            action_ticks = (action_ticks + 1) % 20
            if action_ticks < 8:
                engine.active_button_id = BUTTON_IDS.get(current_action)
            else:
                engine.active_button_id = None

            engine.dll.retro_run()
            frame = engine.get_frame()

            vis_frame = dashboard.render(
                frame, current_scene_probs, current_scene,
                current_action, current_probs, current_latency,
                current_step, current_coords
            )
            cv2.imshow("ClefPlaysPokemon", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - t_frame_start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopping GBA session...")
    finally:
        running = False
        worker.join(timeout=1.0)
        cv2.destroyAllWindows()
        engine.stop()

if __name__ == "__main__":
    main()
