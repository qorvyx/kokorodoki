import os
import queue
import signal
import threading
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

import customtkinter as ctk
from kokoro import KPipeline

from config import (
    MAX_SPEED,
    MIN_SPEED,
    MIN_WINDOW_SIZE,
    READING_FONTS,
    TITLE,
    UI_FONTS,
    VERSION,
    WINDOW_SIZE,
)
from models import TTSPlayer
from utils import (
    OcrReader,
    format_speed,
    get_gui_themes,
    get_language_map,
    get_nltk_language,
    get_voices,
    has_text,
    is_image_file,
    pick_font,
    reading_padding,
    sentence_progress,
    split_text_to_sentences,
)

# "Reading lamp" tokens, as (light, dark) pairs the way CustomTkinter takes them
NIGHT = ("#F6F7FA", "#1B2130")
SURFACE = ("#FFFFFF", "#232B3E")
INK = ("#1D2333", "#E4E7EF")
MUTED = ("#5F6880", "#8B94AA")
MARK = ("#FFD970", "#F2B84B")
# Derived tones for tracks, borders and hovers
TRACK = ("#E3E6EE", "#2F3850")
HOVER = ("#ECEEF3", "#2A3349")
MARK_HOVER = ("#FFE38F", "#F5C76D")
SELECT = ("#D5DBEA", "#3A4562")
ON_MARK = "#1D2333"

PLACEHOLDER = "Type or paste text, or open a file"
# Characters per line in the reading pane
MEASURE = 70
PANE_MIN_PADDING = 28
PANE_PADDING_Y = 24
READING_SIZE = 14


def tone(pair: tuple[str, str], mode: str) -> str:
    """Return the colour of a token in the light or dark mode"""
    return pair[1] if mode == "dark" else pair[0]


class Tooltip:
    """A small hover tip, CustomTkinter has none"""

    def __init__(self, widget, text: str, font: tuple, delay: int = 500):
        self.widget = widget
        self.text = text
        self.font = font
        self.delay = delay
        self.after_id = None
        self.window = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, event=None) -> None:
        self.hide()
        self.after_id = self.widget.after(self.delay, self.show)

    def show(self) -> None:
        mode = ctk.get_appearance_mode().lower()
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(f"+{x}+{y}")
        tk.Label(
            self.window,
            text=self.text,
            font=self.font,
            background=tone(INK, mode),
            foreground=tone(SURFACE, mode),
            padx=8,
            pady=4,
        ).pack()

    def hide(self, event=None) -> None:
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.window is not None:
            self.window.destroy()
            self.window = None


class Gui:
    def __init__(
        self,
        root: ctk.CTk,
        pipeline: KPipeline,
        language: str,
        voice: str,
        speed: float,
        device: str | None,
        mode: str,
    ):
        self.root = root
        self.mode = mode
        self.languages = [lang for _, lang in get_language_map().items()]
        self.voices = get_voices()
        self.current_language_code = language
        self.current_language = get_language_map()[language]
        self.current_voice = voice
        self.speed = speed
        self.device = device
        self.pipeline = pipeline
        self.player = TTSPlayer(
            self.pipeline,
            self.current_language_code,
            self.current_voice,
            self.speed,
            False,
        )
        self.current_thread = None
        # Bumped on each play, so a playback still waiting for the old one can tell it was replaced
        self.play_token = 0
        self.speech_paused = False
        self.loading = False
        self.prev_text = ""
        self.prev_sentences = []
        self.nltk_language = get_nltk_language(self.current_language_code)
        self.sentence_indices = []

        self.reader = OcrReader(language)

        self.queue = queue.Queue()
        self.root.after(100, self.process_queue)

        families = tkfont.families(self.root)
        self.ui_font = pick_font(families, UI_FONTS)
        self.reading_font = tkfont.Font(
            root=self.root, family=pick_font(families, READING_FONTS), size=READING_SIZE
        )
        sample = "the quick brown fox jumps over a lazy dog "
        self.measure = self.reading_font.measure(sample) * MEASURE // len(sample)

        self.create_widgets()
        self.bind_shortcuts()
        self.apply_theme()
        self.update_progress(None)

    def process_queue(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(item, tuple):
                    func, args = item
                    func(*args)
                else:
                    item()
            except Exception as e:
                # Keep polling: later items re-enable the text area.
                print(f"GUI update error: {e}")
        self.root.after(100, self.process_queue)

    def font(self, size: int, weight: str = "normal") -> ctk.CTkFont:
        """Return the UI font at a size in pixels"""
        return ctk.CTkFont(family=self.ui_font, size=size, weight=weight)

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def change_speed(self, speed: float) -> None:
        """Change speed"""
        self.speed = round(speed, 2)
        self.player.change_speed(self.speed)
        self.speed_value.configure(text=format_speed(self.speed))

    def change_voice(self, voice: str) -> None:
        """Change voice"""
        self.current_voice = voice
        self.player.change_voice(self.current_voice)
        self.set_status(f"Voice set to {self.current_voice}")

    def change_lang(self, language: str) -> None:
        """Change language, loading the models in a thread so the window stays responsive"""
        language_code = next(
            code for code, lang in get_language_map().items() if lang == language
        )
        if language_code == self.current_language_code:
            return

        self.set_loading(True)
        self.set_status(f"Loading {language}…")
        threading.Thread(
            target=self.load_language,
            args=(language, language_code),
            daemon=True,
        ).start()

    def load_language(self, language: str, language_code: str) -> None:
        """Load the pipeline for a language, the OCR model loads on first use"""
        try:
            self.player.change_language(language_code, self.device)
            self.reader.set_language(language_code)
        except Exception as e:
            error = str(e)
            self.queue.put(lambda: self.language_failed(language, error))
            return
        self.queue.put(lambda: self.language_loaded(language, language_code))

    def language_loaded(self, language: str, language_code: str) -> None:
        """Update the menus once the new language is loaded"""
        self.current_language = language
        self.current_language_code = language_code
        self.nltk_language = get_nltk_language(language_code)

        self.voice_menu.configure(
            values=[voice for voice in self.voices if voice.startswith(language_code)]
        )
        self.voice_var.set(self.player.voice)
        self.current_voice = self.player.voice

        self.set_loading(False)
        self.set_status(f"Language set to {language}")

    def language_failed(self, language: str, error: str) -> None:
        """Go back to the current language after a failed load"""
        self.lang_var.set(self.current_language)
        self.set_loading(False)
        self.set_status(f"Couldn't load {language}: {error}")

    def set_loading(self, loading: bool) -> None:
        """Disable the menus and transport while a language loads"""
        self.loading = loading
        state = "disabled" if loading else "normal"
        for widget in (
            self.lang_menu,
            self.voice_menu,
            self.back_button,
            self.play_button,
            self.next_button,
        ):
            widget.configure(state=state)
        self.play_button.configure(fg_color=TRACK if loading else MARK)

    def is_playing(self) -> bool:
        return self.current_thread is not None and self.current_thread.is_alive()

    def toggle_playback(self) -> None:
        """Pause while reading, otherwise play or resume"""
        if self.loading:
            return
        if self.is_playing() and not self.speech_paused:
            self.pause_speech()
        else:
            self.play_speech()

    def play_speech(self) -> None:
        """Play or resume if it was paused"""
        text = self.text_area.get("1.0", tk.END).strip()
        if not has_text(text):
            self.set_status("Type or paste some text first")
            return

        is_playing = self.is_playing()
        if self.prev_text == text and self.speech_paused and is_playing:
            self.speech_paused = False
            self.resume_speech()
        else:
            self.speech_paused = False
            self.text_area.delete("1.0", tk.END)
            self.text_area.insert("1.0", text)
            self.prev_text = text
            self.prev_sentences = split_text_to_sentences(text, self.nltk_language)
            # The new thread waits for the old one, so the window doesn't freeze
            previous_thread = self.current_thread
            if is_playing:
                self.player.stop_playback()
            self.play_token += 1
            self.current_thread = threading.Thread(
                target=self.speak_thread,
                args=(self.prev_sentences, previous_thread, self.play_token),
                daemon=True,
            )
            self.current_thread.start()
            self.calculate_sentence_indices()
            self.update_progress(None)
            self.play_button.configure(text="❚❚  Pause")
            self.set_status("Reading…")

    def speak_thread(
        self,
        text: list[str],
        previous_thread: threading.Thread | None,
        token: int,
    ) -> None:
        """Player speak wrapper"""
        try:
            if previous_thread is not None:
                previous_thread.join()
            # Play was pressed again while waiting
            if token != self.play_token:
                return
            self.queue.put(lambda: self.lock_text(True))
            self.player.speak(text, console_mode=False, gui_highlight=self)
        except Exception as e:
            print(f"Error in thread: {str(e)}")
        finally:
            self.queue.put((self.playback_finished, (token,)))

    def playback_finished(self, token: int) -> None:
        """Unlock the text once the latest playback ends"""
        if token != self.play_token:
            return
        self.speech_paused = False
        self.lock_text(False)
        self.play_button.configure(text="▶  Play")
        self.set_status("Finished")

    def lock_text(self, locked: bool) -> None:
        """Make the text read-only while it is read aloud"""
        self.text_area.configure(
            state="disabled" if locked else "normal",
            cursor="arrow" if locked else "xterm",
        )

    def pause_speech(self) -> None:
        """Pause speech"""
        if not self.is_playing():
            return
        self.speech_paused = True
        self.lock_text(False)
        self.player.pause_playback()
        self.play_button.configure(text="▶  Resume")
        self.set_status("Paused")

    def resume_speech(self) -> None:
        """Resume speech"""
        self.lock_text(True)
        self.player.resume_playback()
        self.play_button.configure(text="❚❚  Pause")
        self.set_status("Reading…")

    def calculate_sentence_indices(self) -> None:
        """Calculate start and end indices for each sentence."""
        self.sentence_indices.clear()
        current_pos = 0
        text_content = self.prev_text

        for sentence in self.prev_sentences:
            start_pos = text_content.find(sentence, current_pos)
            if start_pos == -1:
                # Keep positions aligned with sentence numbers.
                self.sentence_indices.append(None)
                continue

            # Calculate line and column for start index
            text_before = text_content[:start_pos]
            start_line = text_before.count("\n") + 1
            start_char = (
                start_pos - text_before.rfind("\n") - 1
                if text_before.rfind("\n") != -1
                else start_pos
            )

            # Calculate end position
            end_pos = start_pos + len(sentence)
            text_before_end = text_content[:end_pos]
            end_line = text_before_end.count("\n") + 1
            end_char = (
                end_pos - text_before_end.rfind("\n") - 1
                if text_before_end.rfind("\n") != -1
                else end_pos
            )

            self.sentence_indices.append(
                {"start": f"{start_line}.{start_char}", "end": f"{end_line}.{end_char}"}
            )
            current_pos = end_pos

    def update_progress(self, sentence: int | None) -> None:
        label, fraction = sentence_progress(sentence, len(self.prev_sentences))
        self.progress_label.configure(text=label)
        # An empty bar still draws a dot of its fill colour
        self.progress_bar.configure(progress_color=MARK if fraction else TRACK)
        self.progress_bar.set(fraction)

    def remove_highlight(self) -> None:
        """Remove highlight"""
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.update_progress(None)

    def highlight(self, sentence: int) -> None:
        """Highlight a sentence"""
        if not self.sentence_indices:
            return

        if not 0 <= sentence < len(self.sentence_indices):
            return
        self.update_progress(sentence)
        indices = self.sentence_indices[sentence]
        if indices is None:
            return
        self.text_area.tag_remove("highlight", "1.0", tk.END)
        self.text_area.tag_add("highlight", indices["start"], indices["end"])
        self.text_area.see(indices["start"])

    def skip_sentence(self) -> None:
        """Skip a sentence"""
        if self.loading or not self.is_playing():
            return
        self.player.skip_sentence()
        self.set_status("Next sentence")

    def back_sentence(self) -> None:
        """Back a sentence"""
        if self.loading or not self.is_playing():
            return
        self.player.back_sentence()
        self.set_status("Back one sentence")

    def toggle_theme(self) -> None:
        """Switch between the dark and light themes"""
        self.mode = "light" if self.mode == "dark" else "dark"
        self.apply_theme()

    def apply_theme(self) -> None:
        """Apply the theme to CustomTkinter and the plain Tk widgets"""
        ctk.set_appearance_mode(self.mode)
        mode = self.mode
        self.text_area.configure(
            background=tone(SURFACE, mode),
            foreground=tone(INK, mode),
            insertbackground=tone(INK, mode),
            selectbackground=tone(SELECT, mode),
            selectforeground=tone(INK, mode),
            inactiveselectbackground=tone(SELECT, mode),
        )
        self.text_area.tag_configure(
            "highlight", background=tone(MARK, mode), foreground=ON_MARK
        )
        self.empty_label.configure(
            background=tone(SURFACE, mode), foreground=tone(MUTED, mode)
        )
        self.theme_button.configure(text="☀" if mode == "dark" else "☾")

    def create_widgets(self) -> None:
        """Create widgets"""
        self.root.configure(fg_color=NIGHT)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        self.add_top_bar()
        self.add_text_area()
        self.add_progress()
        self.add_transport()
        self.add_status_bar()

    def ghost_button(self, master, text: str, command, width: int) -> ctk.CTkButton:
        """A quiet button that only shows its outline"""
        return ctk.CTkButton(
            master,
            text=text,
            command=command,
            width=width,
            height=36,
            corner_radius=18,
            border_width=1,
            border_color=TRACK,
            fg_color="transparent",
            hover_color=HOVER,
            text_color=INK,
            text_color_disabled=MUTED,
            font=self.font(13),
        )

    def add_top_bar(self) -> None:
        """Add the app name, language and voice menus and the theme switch"""
        bar = ctk.CTkFrame(self.root, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=24, pady=(16, 12))
        bar.columnconfigure(0, weight=1)

        ctk.CTkLabel(bar, text=TITLE, font=self.font(16, "bold"), text_color=INK).grid(
            row=0, column=0, sticky="w"
        )

        menu_style = {
            "height": 32,
            "corner_radius": 16,
            "fg_color": SURFACE,
            "button_color": SURFACE,
            "button_hover_color": HOVER,
            "text_color": INK,
            "text_color_disabled": MUTED,
            "dropdown_fg_color": SURFACE,
            "dropdown_hover_color": HOVER,
            "dropdown_text_color": INK,
            "font": self.font(13),
            "dropdown_font": self.font(13),
            "dynamic_resizing": False,
        }

        self.lang_var = tk.StringVar(value=self.current_language)
        self.lang_menu = ctk.CTkOptionMenu(
            bar,
            variable=self.lang_var,
            values=self.languages,
            command=self.change_lang,
            width=176,
            **menu_style,
        )
        self.lang_menu.grid(row=0, column=1, padx=(0, 8))
        Tooltip(self.lang_menu, "Language", self.font(12))

        self.voice_var = tk.StringVar(value=self.current_voice)
        self.voice_menu = ctk.CTkOptionMenu(
            bar,
            variable=self.voice_var,
            values=[
                voice
                for voice in self.voices
                if voice.startswith(self.current_language_code)
            ],
            command=self.change_voice,
            width=136,
            **menu_style,
        )
        self.voice_menu.grid(row=0, column=2, padx=(0, 12))
        Tooltip(self.voice_menu, "Voice", self.font(12))

        self.theme_button = ctk.CTkButton(
            bar,
            text="☀",
            command=self.toggle_theme,
            width=32,
            height=32,
            corner_radius=16,
            fg_color="transparent",
            hover_color=HOVER,
            text_color=MUTED,
            font=self.font(16),
        )
        self.theme_button.grid(row=0, column=3)
        Tooltip(self.theme_button, "Switch between dark and light", self.font(12))

    def add_text_area(self) -> None:
        """Create the reading pane"""
        pane = ctk.CTkFrame(self.root, fg_color=SURFACE, corner_radius=14)
        pane.grid(row=1, column=0, sticky="nsew", padx=24)
        pane.columnconfigure(0, weight=1)
        pane.rowconfigure(0, weight=1)

        self.text_area = tk.Text(
            pane,
            width=1,
            height=1,
            wrap="word",
            font=self.reading_font,
            borderwidth=0,
            highlightthickness=0,
            padx=PANE_MIN_PADDING,
            pady=PANE_PADDING_Y,
            spacing1=2,
            spacing2=7,
            spacing3=12,
            insertwidth=2,
            undo=True,
        )
        # Inside the rounded corners
        self.text_area.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)

        text_scroll = ctk.CTkScrollbar(
            pane,
            command=self.text_area.yview,
            fg_color="transparent",
            button_color=TRACK,
            button_hover_color=MUTED,
            width=12,
        )
        text_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 4), pady=10)
        self.text_area.configure(yscrollcommand=text_scroll.set)

        self.empty_label = tk.Label(
            pane, text=PLACEHOLDER, font=self.reading_font, borderwidth=0
        )
        self.empty_label.bind("<Button-1>", lambda e: self.text_area.focus_set())

        self.text_area.bind("<Configure>", self.fit_measure)
        self.text_area.bind("<<Modified>>", self.text_modified)
        self.text_area.bind(
            "<FocusIn>",
            lambda e: self.set_status("Editing text. Press Esc to use the shortcuts"),
        )
        self.update_empty_state()

    def fit_measure(self, event=None) -> None:
        """Keep the reading text near its measure as the window resizes"""
        padding = reading_padding(
            self.text_area.winfo_width(), self.measure, PANE_MIN_PADDING
        )
        self.text_area.configure(padx=padding)
        self.update_empty_state()

    def text_modified(self, event=None) -> None:
        self.text_area.edit_modified(False)
        self.update_empty_state()

    def update_empty_state(self) -> None:
        """Show the empty state only while the pane has no text"""
        if has_text(self.text_area.get("1.0", "end-1c")):
            self.empty_label.place_forget()
        else:
            # Inside the Text's border, which includes its padding
            self.empty_label.place(in_=self.text_area, x=0, y=2)

    def add_progress(self) -> None:
        """Add the thin progress bar and the sentence count"""
        row = ctk.CTkFrame(self.root, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=(24, 30), pady=(14, 0))
        row.columnconfigure(0, weight=1)

        self.progress_bar = ctk.CTkProgressBar(
            row,
            height=4,
            corner_radius=2,
            fg_color=TRACK,
            progress_color=MARK,
        )
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0, 14))

        self.progress_label = ctk.CTkLabel(
            row, text="", font=self.font(12), text_color=MUTED, width=130, anchor="e"
        )
        self.progress_label.grid(row=0, column=1, sticky="e")

    def add_transport(self) -> None:
        """Add open file, back, play/pause, next and speed"""
        bar = ctk.CTkFrame(self.root, fg_color="transparent")
        bar.grid(row=3, column=0, sticky="ew", padx=24, pady=(8, 0))
        bar.columnconfigure((0, 2), weight=1, uniform="side")

        open_button = self.ghost_button(bar, "Open file", self.choose_file, 104)
        open_button.grid(row=0, column=0, sticky="w")
        Tooltip(open_button, "Open a text file or an image (Ctrl+O)", self.font(12))

        controls = ctk.CTkFrame(bar, fg_color="transparent")
        controls.grid(row=0, column=1)

        self.back_button = self.ghost_button(controls, "⏮", self.back_sentence, 44)
        self.back_button.configure(font=self.font(18))
        self.back_button.grid(row=0, column=0)
        Tooltip(self.back_button, "Back one sentence (←)", self.font(12))

        self.play_button = ctk.CTkButton(
            controls,
            text="▶  Play",
            command=self.toggle_playback,
            width=124,
            height=40,
            corner_radius=20,
            fg_color=MARK,
            hover_color=MARK_HOVER,
            text_color=ON_MARK,
            text_color_disabled=MUTED,
            font=self.font(14, "bold"),
        )
        self.play_button.grid(row=0, column=1, padx=12)
        Tooltip(self.play_button, "Play or pause (Space)", self.font(12))

        self.next_button = self.ghost_button(controls, "⏭", self.skip_sentence, 44)
        self.next_button.configure(font=self.font(18))
        self.next_button.grid(row=0, column=2)
        Tooltip(self.next_button, "Next sentence (→)", self.font(12))

        speed = ctk.CTkFrame(bar, fg_color="transparent")
        speed.grid(row=0, column=2, sticky="e")

        ctk.CTkLabel(speed, text="Speed", font=self.font(13), text_color=MUTED).grid(
            row=0, column=0, padx=(0, 10)
        )
        self.speed_scale = ctk.CTkSlider(
            speed,
            from_=MIN_SPEED,
            to=MAX_SPEED,
            number_of_steps=round((MAX_SPEED - MIN_SPEED) * 10),
            command=self.change_speed,
            width=120,
            height=16,
            fg_color=TRACK,
            progress_color=MUTED,
            button_color=INK,
            button_hover_color=MARK,
        )
        self.speed_scale.set(self.speed)
        self.speed_scale.grid(row=0, column=1)

        self.speed_value = ctk.CTkLabel(
            speed,
            text=format_speed(self.speed),
            font=self.font(13, "bold"),
            text_color=INK,
            width=44,
            anchor="e",
        )
        self.speed_value.grid(row=0, column=2)

    def add_status_bar(self) -> None:
        """Create status bar"""
        bar = ctk.CTkFrame(self.root, fg_color="transparent")
        bar.grid(row=4, column=0, sticky="ew", padx=30, pady=(10, 12))
        bar.columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            bar, text="Ready", font=self.font(12), text_color=MUTED, anchor="w"
        )
        self.status_label.grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(bar, text=VERSION, font=self.font(12), text_color=MUTED).grid(
            row=0, column=1, sticky="e"
        )

    def bind_shortcuts(self) -> None:
        """Space, arrows and Ctrl+O; Space and arrows are left to the pane while editing"""
        for key in ("<Control-o>", "<Control-O>"):
            self.root.bind(key, lambda e: self.choose_file())
            # Tk's Text inserts a line on Ctrl+O
            self.text_area.bind(key, lambda e: (self.choose_file(), "break")[1])
        self.root.bind("<space>", lambda e: self.shortcut(self.toggle_playback))
        self.root.bind("<Left>", lambda e: self.shortcut(self.back_sentence))
        self.root.bind("<Right>", lambda e: self.shortcut(self.skip_sentence))
        self.text_area.bind("<Escape>", lambda e: self.leave_text())

    def shortcut(self, action) -> None:
        """Run a shortcut unless the user is typing in the pane"""
        editing = (
            self.root.focus_get() is self.text_area
            and str(self.text_area.cget("state")) == "normal"
        )
        if not editing:
            action()

    def leave_text(self) -> None:
        self.root.focus_set()
        self.set_status("Space plays or pauses, ← → move by sentence")

    def choose_file(self) -> None:
        """Open a text file, or an image to read with OCR"""
        try:
            file_path = filedialog.askopenfilename(
                title="Open a text file or an image",
                filetypes=[
                    ("Text and images", "*.txt *.md *.srt *.png *.jpg *.jpeg *.webp"),
                    ("All files", "*.*"),
                ],
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            return
        if not file_path:
            return
        if self.is_playing() and not self.speech_paused:
            self.set_status("Pause before opening a file")
            return
        name = os.path.basename(file_path)
        if is_image_file(file_path):
            self.set_status(f"Finding text in {name}…")
            threading.Thread(
                target=self.read_image, args=(file_path,), daemon=True
            ).start()
            return
        try:
            with open(file_path, encoding="utf-8") as file:
                content = file.read()
        except (OSError, UnicodeDecodeError):
            self.set_status(f"Couldn't open {name}")
            return
        self.set_text(content)
        self.set_status(f"Opened {name}")

    def set_text(self, text: str) -> None:
        """Replace the pane's text"""
        self.text_area.delete("1.0", tk.END)
        self.text_area.insert(tk.END, text)

    def read_image(self, file_path: str) -> None:
        """Read text from an image (OCR) outside the UI thread"""
        try:
            results = self.reader.readtext(file_path)
        except Exception as e:
            print(f"OCR error: {e}")
            self.queue.put((self.set_status, ("Couldn't read that image",)))
            return
        image_text = " ".join(text for _, text, _ in results if text).strip()
        self.queue.put((self.show_image_text, (image_text,)))

    def show_image_text(self, image_text: str) -> None:
        """Put the text read from an image in the text area"""
        if self.is_playing() and not self.speech_paused:
            self.set_status("Pause to see the text from the image")
            return
        if not has_text(image_text):
            self.set_status("No text found in that image")
            return
        self.set_text(image_text)
        self.set_status("Text found in the image")

    def close(self):
        if self.current_thread is not None and self.current_thread.is_alive():
            self.player.stop_playback()
            self.current_thread.join()


def setup_signal_handler(root, app):
    """Set up a signal handler for SIGINT (Ctrl+C) to close the Tkinter window."""

    def signal_handler(sig, frame):
        print("\nClosing...")
        app.close()
        root.after(0, root.destroy)

    signal.signal(signal.SIGINT, signal_handler)


def run_gui(
    pipeline: KPipeline,
    language: str,
    voice: str,
    speed: float,
    device: str | None,
    theme: int,
) -> None:
    """Start gui mode"""
    try:
        mode = get_gui_themes()[theme]
        ctk.set_appearance_mode(mode)
        root = ctk.CTk()
        root.title(TITLE)
        root.geometry(WINDOW_SIZE)
        root.minsize(*MIN_WINDOW_SIZE)
        root.resizable(True, True)

        app = Gui(root, pipeline, language, voice, speed, device, mode)

        def on_closing():
            if messagebox.askokcancel("Quit", "Do you want to quit?"):
                print("Exiting...")
                app.close()
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        setup_signal_handler(root, app)

        root.mainloop()

    except Exception as e:
        print(f"An error occurred: {e}")
        messagebox.showerror("Error", f"Application failed to start: {e}")
