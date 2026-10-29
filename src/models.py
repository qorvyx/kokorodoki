import platform
import queue
import sys
import threading
import time

import librosa
import numpy as np
import soundfile as sf
from kokoro import KPipeline
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from config import MAX_SPEED, MIN_SPEED, REPO_ID, SAMPLE_RATE, console
from utils import (
    get_language_map,
    get_nltk_language,
    get_voices,
    parse_srt_file,
    release_memory,
    split_text_to_sentences,
)


class TTSPlayer:
    """Class to handle TTS generation and playback."""

    def __init__(
        self,
        pipeline: KPipeline,
        language: str,
        voice: str,
        speed: float,
        verbose: bool,
        ctrlc: bool = True,
    ):
        self.pipeline = pipeline
        self.language = language
        self.nltk_language = get_nltk_language(self.language)
        self.voice = voice
        self.speed = speed
        self.verbose = verbose
        self.languages = get_language_map()
        self.voices = get_voices()
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.skip = threading.Event()
        self.back = threading.Event()
        self.lock = threading.Lock()
        self.back_number = 0
        self.audio_player = None
        self._threads: list[threading.Thread] = []
        self.ctrlc = not ctrlc
        self.print_complete = True

    def change_language(self, new_lang: str, device: str | None) -> bool:
        """Change the language and reinitialize the pipeline."""
        if new_lang == self.language:
            return True
        if new_lang in self.languages:
            # Load first, so a failed load keeps the current language.
            # The loaded model is shared, only the G2P is new.
            self.pipeline = KPipeline(
                lang_code=new_lang,
                repo_id=REPO_ID,
                model=self.pipeline.model,
                device=device,
            )
            self.language = new_lang
            if not self.voice.startswith(new_lang):
                self.change_voice(
                    next(voice for voice in get_voices() if voice.startswith(new_lang))
                )
            self.nltk_language = get_nltk_language(self.language)
            return True
        return False

    def change_voice(self, new_voice: str) -> bool:
        """Change the voice."""
        if new_voice in self.voices:
            self.voice = new_voice
            return True
        return False

    def change_speed(self, new_speed: float) -> bool:
        """Change the speech speed."""
        if MIN_SPEED <= new_speed <= MAX_SPEED:
            self.speed = new_speed
            return True
        return False

    def trim_silence(self, audio, threshold=0.003):
        """Trim silence from the beginning and end of an audio chunk."""
        # Convert to absolute values to handle negative amplitudes
        abs_audio = np.abs(audio)
        # Find indices where amplitude exceeds threshold
        non_silent = np.where(abs_audio > threshold)[0]

        if len(non_silent) == 0:
            return audio

        # Trim the audio
        start_idx = non_silent[0]
        end_idx = non_silent[-1] + 1
        return audio[start_idx:end_idx]

    def generate_audio(self, text: str | list) -> None:
        """Generate audio chunks and put them in the queue."""
        try:
            sentences = [text] if isinstance(text, str) else text
            for index, sentence in enumerate(sentences):
                generator = self.pipeline(
                    sentence, voice=self.voice, speed=self.speed, split_pattern=None
                )

                for result in generator:
                    if self.stop_event.is_set():
                        self.audio_queue.put(None)
                        return

                    if result.audio is not None:
                        audio = result.audio.numpy()
                        if self.verbose:
                            console.print(
                                f"[dim]Generated: {result.graphemes[:30]}...[/]"
                            )
                        # Trim silence for smooth reading
                        trimed_audio, _ = librosa.effects.trim(audio, top_db=60)
                        # trimed_audio = self.trim_silence(audio, threshold=0.001)
                        # Tag each chunk with its sentence so the GUI highlights
                        # the right one when a sentence yields several chunks.
                        self.audio_queue.put((index, trimed_audio))

            self.audio_queue.put(None)  # Signal end of generation
        except Exception as e:
            console.print(f"[bold red]Generation error:[/] {str(e)}")
            self.audio_queue.put(None)  # Ensure playback thread exits

    def generate_audio_file(self, text: list | str, output_file="Output.wav") -> bool:
        """Generate audio file, return True if it was saved"""
        try:
            with Progress(
                SpinnerColumn("dots", style="yellow", speed=0.8),
                TextColumn("[bold yellow]{task.description}"),
                BarColumn(pulse_style="yellow", complete_style="blue"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    f"[bold yellow]Generating {output_file}",
                    total=None,
                )

                sentences = [text] if isinstance(text, str) else text
                audio_chunks = []
                for sentence in sentences:
                    generator = self.pipeline(
                        sentence, voice=self.voice, speed=self.speed, split_pattern=None
                    )

                    for result in generator:
                        if result.audio is None:
                            continue
                        trimed_audio, _ = librosa.effects.trim(
                            result.audio.numpy(), top_db=70
                        )
                        audio_chunks.append(self.to_stereo(trimed_audio))

                if not audio_chunks:
                    console.print("[bold red]Error:[/] No audio was generated")
                    return False

                full_audio = np.concatenate(audio_chunks, axis=0)
                sf.write(output_file, full_audio, SAMPLE_RATE, format="WAV")

                progress.update(
                    task,
                    completed=1,
                    total=1,
                    description=f"[bold green]Saved to {output_file}[/]",
                )
            return True

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting...[/]")
            sys.exit()
        except Exception as e:
            console.print(f"[bold red]Generation error:[/] {str(e)}")
            return False
        finally:
            release_memory()

    def generate_srt_timed_audio(self, srt_file: str, output_file="Output.wav") -> bool:
        """Generate timed audio based on SRT subtitle file, return True if it was saved"""
        try:
            # Parse SRT file
            srt_entries = parse_srt_file(srt_file)
            if not srt_entries:
                console.print("[bold red]Error:[/] No valid entries found in SRT file")
                return False

            with Progress(
                SpinnerColumn("dots", style="yellow", speed=0.8),
                TextColumn("[bold yellow]{task.description}"),
                BarColumn(pulse_style="yellow", complete_style="blue"),
                TimeElapsedColumn(),
            ) as progress:
                task = progress.add_task(
                    "[bold yellow]Generating timed audio from SRT",
                    total=len(srt_entries),
                )

                # Calculate total duration needed
                total_duration = max(entry.end_time for entry in srt_entries)
                total_samples = int(total_duration * SAMPLE_RATE)

                # Initialize stereo audio array with silence
                full_audio = np.zeros((total_samples, 2))

                for entry in srt_entries:
                    # Split text into sentences for better processing
                    sentences = split_text_to_sentences(entry.text, self.nltk_language)

                    # Generate audio for this entry
                    entry_audio_chunks = []
                    for sentence in sentences:
                        generator = self.pipeline(
                            sentence,
                            voice=self.voice,
                            speed=self.speed,
                            split_pattern=None,
                        )

                        for result in generator:
                            if result.audio is not None:
                                trimmed_audio, _ = librosa.effects.trim(
                                    result.audio.numpy(), top_db=70
                                )
                                entry_audio_chunks.append(self.to_stereo(trimmed_audio))

                    if entry_audio_chunks:
                        # Concatenate all audio for this entry
                        entry_audio = np.concatenate(entry_audio_chunks, axis=0)

                        # Calculate timing
                        start_sample = int(entry.start_time * SAMPLE_RATE)
                        entry_duration = entry.end_time - entry.start_time
                        target_samples = int(entry_duration * SAMPLE_RATE)

                        # Ensure we don't exceed the target duration
                        if len(entry_audio) > target_samples:
                            # If audio is too long, truncate it
                            entry_audio = entry_audio[:target_samples]

                        # Calculate end sample
                        end_sample = start_sample + len(entry_audio)

                        # Make sure we don't exceed the total audio length
                        if end_sample > len(full_audio):
                            end_sample = len(full_audio)
                            entry_audio = entry_audio[: end_sample - start_sample]

                        # Place audio at the correct timing, mixing overlapping entries
                        if start_sample < len(full_audio):
                            full_audio[start_sample:end_sample] += entry_audio

                    progress.update(task, advance=1)

                # Save the final audio
                np.clip(full_audio, -1.0, 1.0, out=full_audio)
                sf.write(output_file, full_audio, SAMPLE_RATE, format="WAV")

                progress.update(
                    task,
                    completed=len(srt_entries),
                    description=f"[bold green]Saved timed audio to {output_file}[/]",
                )
            return True

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Exiting...[/]")
            sys.exit()
        except Exception as e:
            console.print(f"[bold red]SRT generation error:[/] {str(e)}")
            return False
        finally:
            release_memory()

    def to_stereo(self, chunk):
        """Convert mono chunk to stereo"""
        if chunk.ndim == 1:
            return np.stack([chunk, chunk], axis=1)
        if chunk.ndim == 2 and chunk.shape[1] == 2:
            return chunk
        raise ValueError(
            f"Unsupported chunk shape: {chunk.shape}. Expected 1D (mono) or 2D (stereo)."
        )

    def play_audio(self, gui_highlight=None) -> None:
        """Play audio chunks from the queue."""
        if self.audio_player is None:
            self.audio_player = AudioPlayer(SAMPLE_RATE)
        self.audio_player.acquire()
        try:
            audio_chunks = []
            audio_size = 0
            self.back_number = 0
            self.print_complete = True
            stopped = False
            while not self.stop_event.is_set():
                self.skip.clear()
                self.back.clear()
                with self.lock:
                    back_number = self.back_number = min(self.back_number, audio_size)
                if back_number > 0 and audio_size > 0:
                    sentence_index, audio = audio_chunks[audio_size - back_number]
                else:
                    # A timeout re-checks stop_event, since a stop may have
                    # cleared the end-of-generation marker from the queue
                    try:
                        item = self.audio_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    self.audio_queue.task_done()
                    if item is None:
                        break

                    sentence_index, audio = item
                    audio_chunks.append(item)
                    audio_size += 1

                if self.verbose:
                    console.print("[dim]Playing chunk...[/dim]")

                self.audio_player.play(audio)
                if gui_highlight is not None:
                    gui_highlight.queue.put(
                        (gui_highlight.highlight, (sentence_index,))
                    )
                while self.audio_player.is_playing:
                    if self.stop_event.is_set():
                        self.audio_player.stop()
                        stopped = True
                        break
                    elif self.skip.is_set():
                        self.audio_player.stop()
                        break
                    elif self.back.is_set():
                        self.audio_player.stop()
                        break
                    time.sleep(0.2)
                if stopped:
                    break

                with self.lock:
                    if not self.back.is_set() and self.back_number > 0:
                        self.back_number -= 1
            if gui_highlight is not None:
                gui_highlight.queue.put(gui_highlight.remove_highlight)
            self.audio_player.stop()
            if self.print_complete is True and not self.stop_event.is_set():
                console.print("[green]Playback complete.[/]\n")
        except Exception as e:
            console.print(f"[bold red]Playback error:[/] {e}")
            self.audio_player.stop()
        finally:
            self.audio_player.release()

    def is_active(self) -> bool:
        """Whether generation or playback threads from the last speak() are alive."""
        return any(thread.is_alive() for thread in self._threads)

    def wait_until_idle(self, timeout: float) -> bool:
        """Wait for the last speak()'s threads to finish; True if they did."""
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not self.is_active()

    def skip_sentence(self) -> None:
        self.skip.set()

    def back_sentence(self) -> None:
        with self.lock:
            self.back_number += 1
        self.back.set()

    def stop_playback(self, printm=True) -> None:
        """Stop ongoing generation and playback."""
        self.stop_event.set()

        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        if printm:
            console.print("\n[yellow]Playback stopped.[/]\n")

    def pause_playback(self) -> None:
        """Pause playback."""
        if self.audio_player is not None:
            self.audio_player.pause()

    def resume_playback(self) -> None:
        """Resume playback."""
        if self.audio_player is not None:
            self.audio_player.resume()

    def speak(self, text: str | list, console_mode=True, gui_highlight=None) -> None:
        """Start TTS generation and playback in separate threads."""

        self.stop_event.clear()

        # New text always starts unpaused
        self.resume_playback()

        # Make sure the queue is empty
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()

        gen_thread = threading.Thread(
            target=self.generate_audio, args=(text,), daemon=True
        )
        play_thread = threading.Thread(
            target=self.play_audio, args=(gui_highlight,), daemon=True
        )
        self._threads = [gen_thread, play_thread]

        try:
            # Start generation thread
            gen_thread.start()

            # Start playback thread
            play_thread.start()

            # Wait for playback to complete
            play_thread.join()
            gen_thread.join()
        except KeyboardInterrupt:
            if console_mode and self.ctrlc:
                self.stop_playback(False)
                console.print("\n[bold yellow]Interrupted. Type !q to exit.[/]")
                gen_thread.join()
                play_thread.join()
            elif console_mode:
                # self.stop_playback(False)
                self.print_complete = False
                console.print("\n[bold yellow]Type !p to pause.[/]")
            else:
                console.print("\n[bold yellow]Exiting...[/]")
                self.stop_playback(False)
                gen_thread.join()
                play_thread.join()
                sys.exit()
        release_memory()


# Seconds to keep the output device open after playback ends, so back-to-back
# texts don't reopen it.
IDLE_CLOSE_DELAY = 2.0

# Terminating PortAudio on idle drops the JACK client that pipewire-jack keeps
# alive. On Windows, PortAudio's WASAPI backend ties COM setup to the thread
# that initialized it, so there only the stream is closed.
TERMINATE_PORTAUDIO_ON_IDLE = platform.system() != "Windows"


def _portaudio_initialize(sd) -> None:
    """Initialize PortAudio again after _portaudio_terminate (sounddevice internals)."""
    if hasattr(sd, "_initialized") and not sd._initialized:
        sd._initialize()


def _portaudio_terminate(sd) -> None:
    """Shut PortAudio down; skipped if sounddevice's internals are unavailable."""
    if getattr(sd, "_initialized", 0) and hasattr(sd, "_terminate"):
        sd._terminate()


class AudioPlayer:
    """Plays audio clips through a PortAudio stream that is only open while needed.

    sounddevice is imported lazily. Playback sessions (one per play_audio run)
    are counted, and once none is left the stream is closed and PortAudio shut
    down after ``idle_delay`` seconds, so an idle daemon holds no audio device
    (under pipewire-jack, an initialized PortAudio keeps a JACK client alive).
    """

    def __init__(self, samplerate, idle_delay=IDLE_CLOSE_DELAY):
        self.samplerate = samplerate
        self.idle_delay = idle_delay
        self.current_frame = 0
        self.playing = True
        self.event = threading.Event()
        # Guards playback state; taken by the PortAudio callback.
        self.lock = threading.Lock()
        # Serializes opening/closing the stream and session bookkeeping. Never
        # hold self.lock while stopping the stream: stop() waits for the
        # callback, which takes self.lock.
        self.stream_lock = threading.Lock()
        self.current_audio = None
        self.stream = None
        self._sessions = 0
        self._idle_timer = None
        self._idle_token = 0

    def _open_stream(self) -> None:
        """Initialize PortAudio and start the stream if needed (stream_lock held)."""
        if self.stream is not None:
            return
        import sounddevice as sd

        _portaudio_initialize(sd)
        stream = None
        try:
            stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=2,
                callback=self._callback,
                finished_callback=self._finished_callback,
            )
            stream.start()
        except Exception:
            if stream is not None:
                stream.close()
            if TERMINATE_PORTAUDIO_ON_IDLE:
                _portaudio_terminate(sd)
            raise
        self.stream = stream

    def _close_stream(self) -> None:
        """Close the stream and shut PortAudio down (stream_lock held)."""
        if self.stream is None:
            return
        import sounddevice as sd

        stream, self.stream = self.stream, None
        try:
            stream.stop()
            stream.close()
        finally:
            if TERMINATE_PORTAUDIO_ON_IDLE:
                _portaudio_terminate(sd)

    def _cancel_idle_close(self) -> None:
        """Invalidate any pending idle close (stream_lock held)."""
        self._idle_token += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def acquire(self) -> None:
        """Start a playback session; the device stays open until it is released."""
        with self.stream_lock:
            self._sessions += 1
            self._cancel_idle_close()

    def release(self) -> None:
        """End a playback session; the last one schedules the idle close."""
        with self.stream_lock:
            self._sessions = max(0, self._sessions - 1)
            self._cancel_idle_close()
            if self._sessions or self.stream is None:
                return
            token = self._idle_token
            self._idle_timer = threading.Timer(
                self.idle_delay, self._close_if_idle, args=(token,)
            )
            self._idle_timer.daemon = True
            self._idle_timer.start()

    def _close_if_idle(self, token) -> None:
        with self.stream_lock:
            if token != self._idle_token or self._sessions:
                return
            self._idle_timer = None
            # No session is left to play what is loaded, so drop it.
            self.stop()
            self._close_stream()

    def close(self) -> None:
        """Release the audio device now."""
        with self.stream_lock:
            self._cancel_idle_close()
            self._close_stream()

    def _callback(self, outdata, frames, time, status):
        with self.lock:
            if not self.playing or self.current_audio is None:
                outdata.fill(0)
                return

            chunksize = min(len(self.current_audio) - self.current_frame, frames)

            if len(self.current_audio.shape) == 1:
                # Mono audio: copy to all output channels
                for channel in range(outdata.shape[1]):
                    outdata[:chunksize, channel] = self.current_audio[
                        self.current_frame : self.current_frame + chunksize
                    ]
            else:
                # Stereo or multi-channel: copy up to available channels
                channels = min(self.current_audio.shape[1], outdata.shape[1])
                outdata[:chunksize, :channels] = self.current_audio[
                    self.current_frame : self.current_frame + chunksize, :channels
                ]

            if chunksize < frames:
                outdata[chunksize:] = 0
                self.current_audio = None
                self.event.set()

            self.current_frame += chunksize

    def _finished_callback(self):
        """Called when stream is stopped"""
        self.event.set()

    def play(self, audio, blocking=False) -> None:
        """Start playback of a single audio clip"""
        with self.stream_lock:
            self._cancel_idle_close()
            self._open_stream()
            # self.playing is left as is, so a pause also holds the next clip
            with self.lock:
                self.event.clear()
                self.current_audio = audio
                self.current_frame = 0

        if blocking:
            self.event.wait()

    def resume(self) -> None:
        """Resume playback"""
        with self.lock:
            self.playing = True

    def pause(self) -> None:
        """Pause playback"""
        with self.lock:
            self.playing = False

    def stop(self) -> None:
        """Stop playback and clear current audio"""
        with self.lock:
            self.current_frame = 0
            self.current_audio = None
            self.event.set()

    @property
    def is_playing(self) -> bool:
        """Check if audio is actively playing"""
        with self.lock:
            return self.current_audio is not None

    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self.close()
        except Exception:
            pass
