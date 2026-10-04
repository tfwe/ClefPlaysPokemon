import ctypes
import os
import numpy as np
from PIL import Image

CORE_PATH = "/usr/lib/x86_64-linux-gnu/libretro/mgba_libretro.so"

# Standard Libretro Joypad Button IDs
BUTTON_IDS = {
    "B": 0,
    "Y": 1,
    "SELECT": 2,
    "START": 3,
    "UP": 4,
    "DOWN": 5,
    "LEFT": 6,
    "RIGHT": 7,
    "A": 8,
    "X": 9,
    "L": 10,
    "R": 11,
}

class RetroGameInfo(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char_p),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("meta", ctypes.c_char_p),
    ]

# Callback signatures per libretro.h
VIDEO_REFRESH_T = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_size_t)
AUDIO_SAMPLE_T = ctypes.CFUNCTYPE(None, ctypes.c_int16, ctypes.c_int16)
AUDIO_SAMPLE_BATCH_T = ctypes.CFUNCTYPE(ctypes.c_size_t, ctypes.POINTER(ctypes.c_int16), ctypes.c_size_t)
INPUT_POLL_T = ctypes.CFUNCTYPE(None)
INPUT_STATE_T = ctypes.CFUNCTYPE(ctypes.c_int16, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint)
ENVIRONMENT_T = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_uint, ctypes.c_void_p)

class GBAEngine:
    def __init__(self, rom_path: str, core_path: str = CORE_PATH):
        self.BUTTON_IDS = BUTTON_IDS
        if not os.path.exists(core_path):
            raise FileNotFoundError(f"mGBA core missing at {core_path}. Run 'sudo apt install libretro-mgba'.")
        if not os.path.exists(rom_path):
            raise FileNotFoundError(f"ROM file not found at {rom_path}")

        self.dll = ctypes.CDLL(core_path)
        self.active_button_id = None
        self.pixel_format = 0  # 0: 0RGB1555, 1: XRGB8888, 2: RGB565
        self.last_frame = Image.new("RGB", (240, 160))

        # Setup persistent callbacks to avoid garbage collection
        self._env_cb = ENVIRONMENT_T(self._environment_callback)
        self._video_cb = VIDEO_REFRESH_T(self._video_callback)
        self._audio_cb = AUDIO_SAMPLE_T(lambda left, right: None)
        self._audio_batch_cb = AUDIO_SAMPLE_BATCH_T(lambda data, frames: frames)
        self._input_poll_cb = INPUT_POLL_T(lambda: None)
        self._input_state_cb = INPUT_STATE_T(self._input_state_callback)

        self.dll.retro_set_environment(self._env_cb)
        self.dll.retro_set_video_refresh(self._video_cb)
        self.dll.retro_set_audio_sample(self._audio_cb)
        self.dll.retro_set_audio_sample_batch(self._audio_batch_cb)
        self.dll.retro_set_input_poll(self._input_poll_cb)
        self.dll.retro_set_input_state(self._input_state_cb)

        self.dll.retro_init()

        # Load ROM
        with open(rom_path, "rb") as f:
            self._rom_data = f.read()

        game_info = RetroGameInfo(
            path=rom_path.encode("utf-8"),
            data=ctypes.cast(ctypes.c_char_p(self._rom_data), ctypes.c_void_p),
            size=len(self._rom_data),
            meta=None
        )

        self.dll.retro_load_game.argtypes = [ctypes.POINTER(RetroGameInfo)]
        self.dll.retro_load_game.restype = ctypes.c_bool
        if not self.dll.retro_load_game(ctypes.byref(game_info)):
            raise RuntimeError(f"mGBA core failed to load game: {rom_path}")

        # Warm up 60 frames to bypass initial BIOS cycle
        for _ in range(60):
            self.dll.retro_run()

    def _environment_callback(self, cmd: int, data: ctypes.c_void_p) -> bool:
        RETRO_ENVIRONMENT_SET_PIXEL_FORMAT = 10
        if cmd == RETRO_ENVIRONMENT_SET_PIXEL_FORMAT and data:
            self.pixel_format = ctypes.cast(data, ctypes.POINTER(ctypes.c_uint)).contents.value
            return True
        return False

    def _video_callback(self, data: ctypes.c_void_p, width: int, height: int, pitch: int):
        if not data or width == 0 or height == 0:
            return

        # Handle XRGB8888 (32-bit) format
        if self.pixel_format == 1:
            raw = np.ctypeslib.as_array(
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint32)),
                shape=(height, pitch // 4)
            )[:, :width]
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[..., 0] = (raw >> 16) & 0xFF
            rgb[..., 1] = (raw >> 8) & 0xFF
            rgb[..., 2] = raw & 0xFF
            self.last_frame = Image.fromarray(rgb)

        # Handle RGB565 (16-bit) format
        elif self.pixel_format == 2:
            raw = np.ctypeslib.as_array(
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint16)),
                shape=(height, pitch // 2)
            )[:, :width]
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[..., 0] = ((raw >> 11) & 0x1F) * 255 // 31
            rgb[..., 1] = ((raw >> 5) & 0x3F) * 255 // 63
            rgb[..., 2] = (raw & 0x1F) * 255 // 31
            self.last_frame = Image.fromarray(rgb)

        # Handle 0RGB1555 (16-bit) format default
        else:
            raw = np.ctypeslib.as_array(
                ctypes.cast(data, ctypes.POINTER(ctypes.c_uint16)),
                shape=(height, pitch // 2)
            )[:, :width]
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[..., 0] = ((raw >> 10) & 0x1F) * 255 // 31
            rgb[..., 1] = ((raw >> 5) & 0x1F) * 255 // 31
            rgb[..., 2] = (raw & 0x1F) * 255 // 31
            self.last_frame = Image.fromarray(rgb)

    def _input_state_callback(self, port: int, device: int, index: int, id_: int) -> int:
        RETRO_DEVICE_JOYPAD = 1
        if device == RETRO_DEVICE_JOYPAD and port == 0:
            if self.active_button_id is not None and id_ == self.active_button_id:
                return 1
        return 0

    def get_frame(self) -> Image.Image:
        return self.last_frame.copy()

    def send_action(self, button: str, hold_frames: int = 12, release_frames: int = 16):
        self.active_button_id = BUTTON_IDS.get(button)
        for _ in range(hold_frames):
            self.dll.retro_run()

        self.active_button_id = None
        for _ in range(release_frames):
            self.dll.retro_run()
    def get_player_coords(self) -> tuple[int, int]:
        """Returns player (x, y) coordinates from Emerald save block in EWRAM."""
        self.dll.retro_get_memory_data.restype = ctypes.c_void_p
        self.dll.retro_get_memory_data.argtypes = [ctypes.c_uint]
        ewram_ptr = self.dll.retro_get_memory_data(0x101)
        if ewram_ptr:
            try:
                # Emerald USA SaveBlock1 coords location
                x = ctypes.cast(ewram_ptr + 0x2E180, ctypes.POINTER(ctypes.c_int16)).contents.value
                y = ctypes.cast(ewram_ptr + 0x2E182, ctypes.POINTER(ctypes.c_int16)).contents.value
                return (int(x), int(y))
            except Exception:
                pass
        return (0, 0)
    def get_ram_context(self) -> str:
        # Standard GBA EWRAM memory region ID in Libretro is 0x101 (RETRO_MEMORY_SYSTEM_RAM)
        self.dll.retro_get_memory_data.restype = ctypes.c_void_p
        self.dll.retro_get_memory_data.argtypes = [ctypes.c_uint]
        ewram_ptr = self.dll.retro_get_memory_data(0x101)

        if ewram_ptr:
            # Pokémon Emerald USA gBattleTypeFlags is at EWRAM + 0x22B4C
            battle_flag = ctypes.cast(ewram_ptr + 0x22B4C, ctypes.POINTER(ctypes.c_uint8)).contents.value
            if battle_flag != 0:
                return "Status: IN BATTLE. Select moves, use items, or switch. Do not run unless party HP is critical."

        return "Status: OVERWORLD exploration. Navigate Hoenn, enter buildings, talk to NPCs, advance dialog."

    def stop(self):
        self.dll.retro_unload_game()
        self.dll.retro_deinit()
