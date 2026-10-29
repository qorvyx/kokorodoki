import os

from rich.console import Console

SAMPLE_RATE = 24000  # Kokoro-82M sample rate
MIN_SPEED = 0.5
MAX_SPEED = 2.0
DEFAULT_SPEED = 1.2
DEFAULT_LANGUAGE = "a"
DEFAULT_VOICE = "af_heart"
REPO_ID = "hexgrad/Kokoro-82M"
HISTORY_FILE = ".kokorodoki_history"
HISTORY_LIMIT = 1024
PROMPT = "> "
TIMEOUT = 5

COMMANDS = [
    "!lang",
    "!voice",
    "!speed",
    "!stop",
    "!pause",
    "!resume",
    "!next",
    "!back",
    "!status",
    "!list_langs",
    "!list_voices",
    "!list_all_voices",
    "!clear",
    "!clear_history",
    "!verbose",
    "!help",
    "!quit",
    "!ctrlc",
]
HOST = "127.0.0.1"
# The daemon only listens locally; containers that publish the port set
# KOKORODOKI_HOST=0.0.0.0.
BIND_HOST = os.environ.get("KOKORODOKI_HOST", HOST)
PORT = 5600
# MB above which an idle CPU daemon restarts itself (0 disables)
DEFAULT_MAX_MEMORY = 3000
# Seconds the daemon waits for a client to send before dropping it.
CLIENT_TIMEOUT = 5
TITLE = "KokoroDoki"
WINDOW_SIZE = "900x680"
MIN_WINDOW_SIZE = (640, 480)
VERSION = "v0.1.0"
DEFAULT_THEME = 1
# Installed fonts are picked at runtime, first match wins
UI_FONTS = ["Inter", "Segoe UI", "Cantarell", "Noto Sans", "DejaVu Sans"]
READING_FONTS = ["Noto Serif", "Georgia", "DejaVu Serif"]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

console = Console()
