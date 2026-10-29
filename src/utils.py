import ctypes
import gc
import os
import platform
import re
import socket
import threading
from dataclasses import dataclass

if platform.system() == "Windows":
    import pyreadline3 as readline
else:
    import readline

from nltk import sent_tokenize
from rich import box
from rich.table import Table

from config import COMMANDS, HISTORY_FILE, HISTORY_LIMIT, IMAGE_EXTENSIONS, console


def read_client_message(conn: socket.socket, timeout: float) -> bytes | None:
    """Read one daemon message until the client closes; None if it goes silent"""
    conn.settimeout(timeout)
    chunks = []
    try:
        while chunk := conn.recv(4096):
            chunks.append(chunk)
    except TimeoutError:
        return None
    return b"".join(chunks)


def is_client_message(data: bytes) -> bool:
    """Whether data looks like a client payload: TEXT:, IMAGE: or plain text"""
    if data.startswith((b"TEXT:", b"IMAGE:")):
        return True
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return bool(text.strip()) and all(c.isprintable() or c.isspace() for c in text)


def is_command_message(data: bytes) -> bool:
    """Whether data is a bare !command; TEXT:/IMAGE: payloads are always read aloud"""
    if data.startswith((b"TEXT:", b"IMAGE:")):
        return False
    return data.decode(errors="replace").startswith("!")


def get_language_map() -> dict[str, str]:
    """Return the available languages"""
    return {
        "a": "American English",
        "b": "British English",
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "p": "Brazilian Portuguese",
        "j": "Japanese",
        "z": "Mandarin Chinese",
    }


def get_easyocr_language_map() -> dict[str, str]:
    """Return the available languages for EasyOCR"""
    return {
        "a": "en",
        "b": "en",
        "e": "es",
        "f": "fr",
        "h": "hi",
        "i": "it",
        "p": "pt",
        "j": "ja",
        "z": "ch_sim",
    }


def get_easyocr_languages(language_code: str) -> list[str]:
    """Return the EasyOCR languages for a language code"""
    lang = get_easyocr_language_map().get(language_code)
    return [lang] if lang else []


def ocr_needs_load(loaded: list[str] | None, wanted: list[str]) -> bool:
    """Whether the OCR reader must be (re)built for the wanted languages"""
    return loaded != wanted


class OcrReader:
    """EasyOCR reader loaded on first use, rebuilt only when its languages change"""

    def __init__(self, language_code: str):
        self.languages = get_easyocr_languages(language_code)
        self._loaded: list[str] | None = None
        self._reader = None
        self._lock = threading.Lock()

    def set_language(self, language_code: str) -> None:
        """Use a new language, loaded on the next read"""
        with self._lock:
            self.languages = get_easyocr_languages(language_code)

    def load(self):
        """Load the reader if needed and return it"""
        with self._lock:
            if ocr_needs_load(self._loaded, self.languages):
                import easyocr

                # Drop the old reader first, so two are never held at once
                self._reader = self._loaded = None
                self._reader = easyocr.Reader(self.languages)
                self._loaded = self.languages
            return self._reader

    def readtext(self, image) -> list:
        """Read text from an image path or bytes"""
        return self.load().readtext(image)


def parse_rss_mb(status: str) -> float | None:
    """Return VmRSS in MB from a /proc/<pid>/status text"""
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) / 1024
    return None


def get_rss_mb() -> float | None:
    """Return this process's resident memory in MB, None if unknown"""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            return parse_rss_mb(f.read())
    except OSError:
        return None


def over_memory_limit(rss_mb: float | None, limit_mb: int) -> bool:
    """Whether rss is known and above a limit; a limit of 0 disables it"""
    return limit_mb > 0 and rss_mb is not None and rss_mb > limit_mb


def restart_argv(argv: list[str], language: str, voice: str, speed: float) -> list[str]:
    """Return argv with the current language, voice and speed, for a restart"""
    options = {"--language", "-l", "--voice", "-v", "--speed", "-s"}
    kept = []
    skip_value = False
    for arg in argv:
        if skip_value:
            skip_value = False
        elif arg in options:
            skip_value = True
        elif arg.split("=", 1)[0] not in options:
            kept.append(arg)
    return [*kept, "--language", language, "--voice", voice, "--speed", str(speed)]


_malloc_trim = None


def release_memory() -> None:
    """Give freed memory back to the OS (glibc only, no-op elsewhere)"""
    global _malloc_trim
    gc.collect()
    if platform.system() != "Linux":
        return
    if _malloc_trim is None:
        try:
            _malloc_trim = ctypes.CDLL("libc.so.6").malloc_trim
        except (OSError, AttributeError):
            # Not glibc (e.g. musl)
            _malloc_trim = False
    if _malloc_trim:
        _malloc_trim(0)


def get_voices() -> list[str]:
    """Return the available voices"""
    return [
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ]


def get_default_voice(language: str, default: str) -> str:
    """Return default if it suits the language, else the language's first voice"""
    if default.startswith(language):
        return default
    return next(voice for voice in get_voices() if voice.startswith(language))


def get_gui_themes() -> dict[int, str]:
    """Return the gui theme for each theme number, old numbers kept working"""
    return {
        1: "dark",
        2: "dark",
        3: "dark",
        4: "dark",
        5: "light",
        6: "light",
        7: "light",
    }


def display_themes() -> None:
    """Display available gui themes"""
    table = Table(title="Available Themes", box=box.ROUNDED)
    table.add_column("Number", style="cyan")
    table.add_column("Theme", style="green")

    table.add_row("1 (or 2-4)", "Dark")
    table.add_row("5 (or 6-7)", "Light")

    console.print(table)


def pick_font(available: list[str] | tuple[str, ...], preferred: list[str]) -> str:
    """Return the first preferred font family that is installed, else the last one"""
    installed = {family.lower(): family for family in available}
    for family in preferred:
        if family.lower() in installed:
            return installed[family.lower()]
    return preferred[-1]


def has_text(text: str) -> bool:
    """Whether the reading pane holds something to read"""
    return bool(text.strip())


def is_image_file(path: str) -> bool:
    """Whether a path looks like an image to OCR"""
    return os.path.splitext(path.lower())[1] in IMAGE_EXTENSIONS


def sentence_progress(index: int | None, total: int) -> tuple[str, float]:
    """Return the progress label and fraction for the sentence being read"""
    if total <= 0:
        return "", 0.0
    if index is None:
        return f"{total} sentence{'' if total == 1 else 's'}", 0.0
    index = max(0, min(index, total - 1))
    return f"Sentence {index + 1} of {total}", (index + 1) / total


def format_speed(speed: float) -> str:
    """Format a speed for the gui, e.g. 1.2×"""
    return f"{round(speed, 2):g}×"


def reading_padding(width: int, measure: int, minimum: int) -> int:
    """Return the side padding that keeps the reading text near its measure"""
    return max(minimum, (width - measure) // 2)


def get_nltk_language_map() -> dict[str, str]:
    """Return available languages in nltk"""
    return {
        "a": "english",
        "b": "english",
        "e": "spanish",
        "f": "french",
        "i": "italian",
        "p": "portuguese",
    }


def get_nltk_language(language_code: str) -> str:
    return next(
        (
            lang
            for code, lang in get_nltk_language_map().items()
            if code == language_code
        ),
        "english",
    )


def display_languages() -> None:
    """Display available languages in a formatted table."""
    languages = get_language_map()
    table = Table(title="Available Languages", box=box.ROUNDED)
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="green")

    for code, name in languages.items():
        table.add_row(code, name)

    console.print(table)


def display_voices(language=None) -> None:
    """Display available voices in a formatted table."""
    voices = get_voices()
    table = Table(title="Available Voices", box=box.ROUNDED)
    table.add_column("Voice ID", style="cyan")
    table.add_column("Prefix", style="yellow")

    if language not in get_language_map() and language is not None:
        console.print(f"[bold red]Error:[/] Invalid language '{language}'")
        display_languages()
        return

    for voice in voices:
        prefix, _ = voice.split("_", 1)
        if language is None or language == prefix[0]:
            prefix_desc = {
                "a": "American",
                "b": "British",
                "e": "Spanish",
                "f": "French",
                "h": "Hindi",
                "i": "Italian",
                "p": "Portuguese",
                "j": "Japanese",
                "z": "Mandarin",
            }.get(prefix[0], "Unknown")
            gender = "Female" if prefix[1] == "f" else "Male"
            table.add_row(voice, f"{prefix_desc} {gender}")

    console.print(table)


def display_status(language: str, voice: str, speed: float) -> None:
    """Display current settings."""
    console.print("[green]Currently:[/]")
    console.print(f"  Language: [cyan]{get_language_map()[language]}[/]")
    console.print(f"  Voice: [cyan]{voice}[/]")
    console.print(f"  Speed: [cyan]{speed}[/]")


def format_status(language: str, voice: str, speed: float) -> str:
    """Return current settings as a formatted string."""
    status_lines = [
        f"Current language: {get_language_map()[language]}.",
        f"Current voice: {voice}.",
        f"Current speed: {speed}.",
    ]
    return "\n".join(status_lines)


def display_help() -> None:
    """Display help information for available commands."""
    table = Table(
        title="[bold]Command Help[/bold]",
        box=box.ROUNDED,
        title_style="bold magenta",
        header_style="bold white",
    )
    table.add_column("Command", style="cyan", width=20)
    table.add_column("Description", style="green")
    table.add_column("Example", style="yellow")

    command_groups = [
        {
            "group": "Playback Control",
            "commands": [
                ("!stop, !s", "Stop current playback", "!stop"),
                ("!pause, !p", "Pause playback", "!pause"),
                ("!resume, !r", "Resume playback", "!resume"),
                ("!next, !n", "Skip to next sentence", "!next"),
                ("!back, !b", "Go to previous sentence", "!back"),
            ],
        },
        {
            "group": "Audio Settings",
            "commands": [
                ("!lang <code>", "Change language", "!lang b"),
                ("!voice <name>", "Change voice", "!voice af_bella"),
                ("!speed <value>", "Set playback speed (0.5-2.0)", "!speed 1.5"),
            ],
        },
        {
            "group": "Information",
            "commands": [
                ("!list_langs", "List available languages", "!list_langs"),
                (
                    "!list_voices",
                    "List available voices for current language",
                    "!list_voices",
                ),
                (
                    "!list_all_voices",
                    "List voices for all languages",
                    "!list_all_voices",
                ),
                ("!status", "Show current settings", "!status"),
            ],
        },
        {
            "group": "Interface",
            "commands": [
                ("!clear", "Clear the screen", "!clear"),
                ("!clear_history", "Clear command history", "!clear_history"),
                ("!verbose", "Toggle verbose mode", "!verbose"),
                ("!ctrlc", "Change the effect of Ctrl+C", "!ctrlc"),
                ("!help, !h", "Show this help message", "!help"),
                ("!quit, !q", "Exit the program", "!quit"),
            ],
        },
    ]

    for group in command_groups:
        table.add_row(
            f"[bold underline]{group['group']}[/]", "", "", style="bold green"
        )
        for cmd, desc, example in group["commands"]:
            table.add_row(cmd, desc, example)

    console.print(table)
    console.print(
        "[italic dim]Tip: Use commands with aliases (e.g., !s for !stop) for faster input.[/]\n"
    )


def clear_history() -> None:
    readline.clear_history()
    if platform.system() != "Windows":
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            pass
    console.print("[bold yellow]History cleared.[/]")


def save_history(history_off: bool) -> None:
    if platform.system() == "Windows":
        return
    if not history_off:
        try:
            readline.write_history_file(HISTORY_FILE)
        except OSError:
            console.print("[bold red]Error saving history file.[/]")


def init_history(history_off: bool) -> None:
    """Load history file and set the limit"""
    if platform.system() == "Windows":
        return
    if not history_off:
        if os.path.exists(HISTORY_FILE):
            try:
                readline.read_history_file(HISTORY_FILE)
            except OSError:
                pass
        readline.set_history_length(HISTORY_LIMIT)


def init_completer() -> None:
    if platform.system() == "Windows":
        return
    readline.parse_and_bind("tab: complete")
    readline.set_completer(completer)
    readline.set_completer_delims("")
    readline.parse_and_bind("set completion-ignore-case on")


def completer(text: str, state: int) -> str | None:
    """Auto-complete function for readline."""
    if platform.system() == "Windows":
        return None
    options = [cmd for cmd in COMMANDS if cmd.startswith(text)]
    return options[state] if state < len(options) else None


def split_by_words(chunk: str, max_len: int) -> list[str]:
    """Split a chunk into smaller chunks by words, ensuring each is <= max_len"""

    if len(chunk) <= max_len:
        return [chunk]

    chunks = []
    current_chunk = ""

    words = chunk.split(" ")
    for word in words:
        if len(current_chunk) + len(word) + (1 if current_chunk else 0) > max_len:
            if current_chunk:
                chunks.append(current_chunk)
            # If word itself is too long
            while len(word) > max_len:
                chunks.append(word[:max_len])
                word = word[max_len:]
            current_chunk = word
        else:
            current_chunk += (" " if current_chunk else "") + word
    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def split_long_sentence(sentence: str, max_len=350, min_len=50) -> list[str]:
    """Split a sentence into chunks of <= max_len"""
    if len(sentence) <= max_len:
        return [sentence]

    chunks = []

    # Split by newline
    chunk = sentence
    if len(chunk) <= max_len:
        chunks.append(chunk)
    else:
        # Split after "[,;]\s" or a CJK comma
        split_points = [m.end() for m in re.finditer(r"[,;]\s|[、，；]", chunk)]
        if split_points:
            last_pos = 0
            for next_pos in split_points:
                segment = chunk[last_pos:next_pos]
                if len(segment) > max_len:
                    # For too long segments
                    chunks.extend(split_by_words(segment, max_len))
                elif len(segment) < min_len and last_pos > 0:
                    # Merge short segment with previous chunk
                    chunks[-1] += segment
                else:
                    chunks.append(segment)
                last_pos = next_pos
            # Handle remaining text
            if last_pos < len(chunk):
                chunks.extend(split_by_words(chunk[last_pos:], max_len))
        else:
            chunks.extend(split_by_words(chunk, max_len))

    for i, chunk in enumerate(chunks):
        if len(chunk) > max_len:
            chunks[i : i + 1] = split_by_words(chunk, max_len)
    return [chunk for chunk in chunks if chunk]


def merge_short_sentences(sentences: list[str], min_len=50, max_len=300) -> list[str]:
    """Merge those shorter than min_len with the next sentence"""
    if not sentences:
        return []

    result = []
    current_sentence = sentences[0]

    for i in range(1, len(sentences)):
        if len(current_sentence) < min_len and len(sentences[i]) < max_len:
            current_sentence += sentences[i]
        else:
            result.append(current_sentence)
            current_sentence = sentences[i]

    if current_sentence:
        result.append(current_sentence)

    return result


# Sentence ends that nltk's punkt models don't know (Japanese, Mandarin, Hindi).
NON_LATIN_SENTENCE_END = re.compile(r"(?<=[。！？।॥])")

# Max sentence length in characters. Text without spaces (Japanese, Mandarin)
# packs several phonemes into each character, and Kokoro drops whatever goes
# past 510 phonemes, so it gets a smaller limit.
MAX_SENTENCE_LEN = 350
MAX_SENTENCE_LEN_NO_SPACES = 100


def split_text_to_sentences(text: str, language: str) -> list[str]:
    """Tokenize text into sentences"""
    sentences = [
        part.strip()
        for sentence in sent_tokenize(text, language=language)
        for part in NON_LATIN_SENTENCE_END.split(sentence)
        if part.strip()
    ]

    new_sentences = []
    for sentence in sentences:
        max_len = MAX_SENTENCE_LEN if " " in sentence else MAX_SENTENCE_LEN_NO_SPACES
        if len(sentence) > max_len:
            new_sentences.extend(split_long_sentence(sentence, max_len=max_len))
        else:
            new_sentences.extend([sentence])

    # new_sentences = merge_short_sentences(new_sentences, min_len=50, max_len=300)
    return new_sentences


@dataclass
class SRTEntry:
    """Represents a single SRT subtitle entry"""

    index: int
    start_time: float  # in seconds
    end_time: float  # in seconds
    text: str


def parse_srt_timestamp(timestamp: str) -> float:
    """Parse SRT timestamp format (HH:MM:SS,mmm) to seconds"""
    # Replace comma with dot for milliseconds
    timestamp = timestamp.replace(",", ".")
    parts = timestamp.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_ms = float(parts[2])

    return hours * 3600 + minutes * 60 + seconds_ms


def parse_srt_file(file_path: str) -> list[SRTEntry]:
    """Parse an SRT subtitle file and return a list of SRTEntry objects"""
    entries = []

    with open(file_path, encoding="utf-8-sig") as f:
        content = f.read().strip()

    # Split by double newlines to separate entries
    blocks = re.split(r"\n\s*\n", content)

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        try:
            # Parse index
            index = int(lines[0])

            # Parse timestamp line
            timestamp_line = lines[1]
            timestamp_match = re.match(
                r"(\d+:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d+:\d{2}:\d{2}[,.]\d{3})",
                timestamp_line,
            )
            if not timestamp_match:
                continue

            start_time = parse_srt_timestamp(timestamp_match.group(1))
            end_time = parse_srt_timestamp(timestamp_match.group(2))
            if end_time < start_time:
                console.print(
                    f"[yellow]Skipping SRT entry {index}: it ends before it starts[/]"
                )
                continue

            # Join text lines (in case subtitle spans multiple lines)
            text = "\n".join(lines[2:]).strip()

            entries.append(SRTEntry(index, start_time, end_time, text))

        except (ValueError, IndexError):
            # Skip malformed entries
            continue

    return entries
