import os
import platform
import queue
import socket
import sys
import threading
import warnings

from config import (
    BIND_HOST,
    CLIENT_TIMEOUT,
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_MEMORY,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    MAX_SPEED,
    MIN_SPEED,
    PROMPT,
    REPO_ID,
    TIMEOUT,
    console,
)

with console.status(
    "[yellow]Initializing Kokoro...[/]",
    spinner="dots",
    spinner_style="yellow",
    speed=0.8,
):
    # Ignoring the following warnings:
    # 1. RNN dropout warning (UserWarning) - Expected behavior when dropout > 0 with num_layers=1
    # 2. weight_norm deprecation (FutureWarning) - Still functional, will update when PyTorch removes it
    # These are safe to suppress as they don't affect model behavior.
    warnings.filterwarnings(
        "ignore",
        message="dropout option adds dropout after all but last recurrent layer",
        category=UserWarning,
        module="torch.nn.modules.rnn",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.nn\.utils\.weight_norm` is deprecated",
        category=FutureWarning,
        module="torch.nn.utils.weight_norm",
    )
    from kokoro import KPipeline
console.print("[bold green]Kokoro initialized!")

import nltk

from input_hander import Args, get_input
from models import TTSPlayer
from utils import (
    OcrReader,
    clear_history,
    display_help,
    display_languages,
    display_status,
    display_voices,
    format_status,
    get_language_map,
    get_rss_mb,
    get_voices,
    is_client_message,
    is_command_message,
    over_memory_limit,
    read_client_message,
    restart_argv,
    split_text_to_sentences,
)


def stop_and_wait(player: TTSPlayer) -> None:
    """Stop the player's last playback and wait for its threads to finish."""
    if player.is_active():
        player.stop_playback(False)
        if not player.wait_until_idle(TIMEOUT):
            console.print(
                "[red]Warning: Threads still active after timeout, proceeding anyway.[/]"
            )


def start(args: Args) -> None:
    """Initialize and run"""
    try:
        with console.status(
            "[yellow]Initializing Kokoro pipeline...[/]",
            spinner="dots",
            spinner_style="yellow",
            speed=0.8,
        ):
            # Initialize TTS pipeline
            pipeline = KPipeline(
                lang_code=args.language, repo_id=REPO_ID, device=args.device
            )
        console.print("[bold green]Kokoro pipeline initialized!")

        # Download nltk tokenizers if not found
        try:
            nltk.data.find("tokenizers/punkt")
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            with console.status(
                "[yellow]Download nltk tokenizers...[/]",
                spinner="dots",
                spinner_style="yellow",
                speed=0.8,
            ):
                nltk.download("punkt", quiet=True)
                nltk.download("punkt_tab", quiet=True)
            console.print("[bold green]Downloading nltk tokenizers finished!")

        # audio_warmup()

        if args.setup:
            # Pre-download the OCR model, which is otherwise loaded on first use
            OcrReader(args.language).load()
            return
        elif args.daemon:
            run_daemon(
                pipeline,
                args.language,
                args.voice,
                args.speed,
                args.device,
                args.verbose,
                args.port,
                get_max_memory(args.max_memory, pipeline),
            )
        elif args.gui:
            from gui import run_gui

            run_gui(
                pipeline,
                args.language,
                args.voice,
                args.speed,
                args.device,
                args.theme,
            )
        elif args.all_voices and args.input_text:
            run_with_all(
                pipeline, args.language, args.speed, args.verbose, args.input_text
            )
        elif args.input_text and args.is_srt_file:
            run_srt_cli(
                pipeline,
                args.language,
                args.voice,
                args.speed,
                args.verbose,
                args.input_text,  # This contains the SRT file path
                args.output_file,
            )
        elif args.input_text:
            run_cli(
                pipeline,
                args.language,
                args.voice,
                args.speed,
                args.verbose,
                args.input_text,
                args.output_file,
            )
        else:
            run_console(
                pipeline,
                args.language,
                args.voice,
                args.speed,
                args.verbose,
                args.history_off,
                args.device,
                args.ctrl_c,
                PROMPT,
            )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Terminated[/]")
    except EOFError:
        console.print("\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")


def get_max_memory(max_memory: int | None, pipeline: KPipeline) -> int:
    """Return the daemon's memory limit in MB, the default only applies on CPU"""
    if max_memory is not None:
        return max_memory
    return DEFAULT_MAX_MEMORY if pipeline.model.device.type == "cpu" else 0


def restart_daemon(server_socket: socket.socket, player: TTSPlayer) -> None:
    """Re-execute the daemon with its arguments and current settings to free memory"""
    server_socket.close()
    sys.stdout.flush()
    argv = restart_argv(sys.argv, player.language, player.voice, player.speed)
    os.execv(sys.executable, [sys.executable, *argv])


def speak_thread(clipboard_data: str, player: TTSPlayer) -> None:
    """Player speak wrapper"""
    try:
        player.speak(clipboard_data, console_mode=False)
    except Exception as e:
        print(f"Error in thread: {str(e)}")


def read_client(
    conn: socket.socket, addr: tuple, messages: "queue.Queue[bytes]"
) -> None:
    """Read one message in its own thread so a silent client can't block others"""
    with conn:
        try:
            data = read_client_message(conn, CLIENT_TIMEOUT)
        except OSError as e:
            print(f"Dropped {addr}: {e}")
            return
    if data is None:
        print(f"Dropped {addr}: no data within {CLIENT_TIMEOUT}s")
    elif not is_client_message(data):
        print(f"Dropped {addr}: unrecognized {len(data)}-byte message")
    else:
        messages.put(data)


def accept_clients(
    server_socket: socket.socket, messages: "queue.Queue[bytes]"
) -> None:
    """Accept connections forever, handing each to a reader thread"""
    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        threading.Thread(
            target=read_client, args=(conn, addr, messages), daemon=True
        ).start()


def run_daemon(
    pipeline: KPipeline,
    language: str,
    voice: str,
    speed: float,
    device: str | None,
    verbose: bool,
    port: int,
    max_memory: int,
) -> None:
    """Start daemon mode"""
    current_thread = None
    # Only restart once something was read, so a low limit can't loop
    has_spoken = False
    player = TTSPlayer(pipeline, language, voice, speed, verbose)
    image_reader = OcrReader(language)

    if max_memory > 0 and get_rss_mb() is None:
        print("Memory usage can't be read here, --max-memory is ignored.")
        max_memory = 0
    elif max_memory > 0:
        print(f"Restarting when idle above {max_memory} MB")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            # Lets a restarted daemon bind again at once (on Windows it would
            # let other processes take the port)
            if platform.system() != "Windows":
                server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((BIND_HOST, port))
            server_socket.listen(5)
            print(f"Listening on {BIND_HOST}:{port}...")

            # Messages are handled here, one at a time, in arrival order.
            messages: queue.Queue[bytes] = queue.Queue()
            threading.Thread(
                target=accept_clients, args=(server_socket, messages), daemon=True
            ).start()

            while True:
                try:
                    # A timeout keeps Ctrl+C responsive on Windows.
                    data = messages.get(timeout=1)
                except queue.Empty:
                    idle = current_thread is None or not current_thread.is_alive()
                    if (
                        has_spoken
                        and idle
                        and over_memory_limit(get_rss_mb(), max_memory)
                    ):
                        print(f"Memory above {max_memory} MB while idle, restarting...")
                        restart_daemon(server_socket, player)
                    continue
                try:
                    if data.startswith(b"IMAGE:"):
                        results = image_reader.readtext(data[6:])
                        clipboard_data = ""
                        clipboard_data = " ".join(
                            text for _, text, _ in results if text
                        ).strip()
                        if not clipboard_data:
                            continue
                    elif data.startswith(b"TEXT:"):
                        clipboard_data = data[5:].decode(errors="replace")
                    else:
                        clipboard_data = data.decode(errors="replace")

                    print(f"Received {clipboard_data[:20]}...")

                    # Handle commands (clipboard text starting with "!" is read aloud)
                    if is_command_message(data):
                        parts = clipboard_data.split(maxsplit=1)
                        cmd = parts[0].lower()
                        arg = parts[1] if len(parts) > 1 else ""

                        if cmd == "!lang":
                            if player.change_language(arg, device):
                                print(f"Language changed to: {player.languages[arg]}")
                                image_reader.set_language(arg)
                            else:
                                print("Invalid language code.")

                        elif cmd == "!voice":
                            if player.change_voice(arg):
                                print(f"Voice changed to: {arg}")
                            else:
                                print("Invalid voice.")

                        elif cmd == "!speed":
                            try:
                                new_speed = float(arg)
                                if player.change_speed(new_speed):
                                    print(f"Speed changed to: {new_speed}")
                                else:
                                    print(
                                        f"Speed must be between {MIN_SPEED} and {MAX_SPEED}"
                                    )
                            except ValueError:
                                print("Invalid speed value")

                        elif cmd == "!pause":
                            player.pause_playback()
                        elif cmd == "!resume":
                            player.resume_playback()
                        elif cmd == "!back":
                            player.back_sentence()
                        elif cmd == "!next":
                            player.skip_sentence()
                        elif cmd in ("!stop", "!exit", "!status"):
                            if current_thread is not None and current_thread.is_alive():
                                print("Stopping previous playback...")
                                player.stop_playback()
                                current_thread.join()
                            if cmd == "!exit":
                                print("Exiting...")
                                if (
                                    current_thread is not None
                                    and current_thread.is_alive()
                                ):
                                    print("Stopping previous playback...")
                                    player.stop_playback()
                                    current_thread.join()
                                sys.exit(0)
                            if cmd == "!status":
                                status_str = format_status(
                                    player.language, player.voice, player.speed
                                )
                                current_thread = threading.Thread(
                                    target=speak_thread,
                                    args=(status_str, player),
                                )
                                current_thread.daemon = True
                                current_thread.start()
                                has_spoken = True
                    else:
                        if current_thread is not None and current_thread.is_alive():
                            print("Stopping previous playback...")
                            player.stop_playback()
                            current_thread.join()
                        sentences = split_text_to_sentences(
                            clipboard_data, player.nltk_language
                        )
                        current_thread = threading.Thread(
                            target=speak_thread,
                            args=(sentences, player),
                        )
                        current_thread.daemon = True
                        current_thread.start()
                        has_spoken = True
                        print("Started new playback thread")
                except Exception as e:
                    # One bad message must not take the daemon down.
                    print(f"Error handling message: {e}")

    except KeyboardInterrupt:
        print("Exiting...")
        if current_thread is not None and current_thread.is_alive():
            player.stop_playback()
            current_thread.join(timeout=1)
        sys.exit()
    except Exception as e:
        print(f"Error: {str(e)}")
        if current_thread is not None and current_thread.is_alive():
            player.stop_playback()
            current_thread.join(timeout=1)
        try:
            if "Address already in use" in str(e):
                print(f"Error: Port {port} is already in use.")
                print("This could be due to:")
                print("  - Another instance of this program running.")
                print(f"  - A different process using port {port}.")
                print("To resolve this:")
                print(
                    "  - Check for and terminate any other instances of this program."
                )
                print(
                    "  - Alternatively, use a different port with the --port option (e.g., --port 9911)."
                )
            run_cli(
                pipeline,
                DEFAULT_LANGUAGE,
                DEFAULT_VOICE,
                DEFAULT_SPEED,
                False,
                f"Error: {str(e)[:40]}. for more info see logs. Exiting.",
                None,
            )
        except Exception as e:
            pass


def run_with_all(
    pipeline: KPipeline,
    language: str,
    speed: float,
    verbose: bool,
    input_text: str,
) -> None:
    """Run with all available voices"""
    console.print(
        f"\n[bold blue]Reading with all available {get_language_map()[language]} voices[/]\n"
    )
    target_voices = [voice for voice in get_voices() if voice.startswith(language)]

    player = TTSPlayer(pipeline, language, target_voices[0], speed, verbose)
    sentences = split_text_to_sentences(input_text, player.nltk_language)
    try:
        for voice in target_voices:
            player.change_voice(voice)
            console.print(f"[cyan]{voice} speaking:[/] {input_text[:30]}")
            player.speak(sentences, console_mode=False)
    except KeyboardInterrupt:
        console.print("[bold yellow]Exiting...[/]")
        stop_and_wait(player)
        sys.exit()


def run_cli(
    pipeline: KPipeline,
    language: str,
    voice: str,
    speed: float,
    verbose: bool,
    input_text: str,
    output_file: str | None,
) -> None:
    """Generate audio"""
    player = TTSPlayer(pipeline, language, voice, speed, verbose)
    sentences = split_text_to_sentences(input_text, player.nltk_language)
    if output_file is None:
        try:
            with console.status(
                f"[cyan]Speaking:[/] {input_text[:30]}...", spinner_style="cyan"
            ):
                player.speak(sentences, console_mode=False)
        except KeyboardInterrupt:
            console.print("[bold yellow]Exiting...[/]")
            stop_and_wait(player)
            sys.exit()
    elif not player.generate_audio_file(sentences, output_file=output_file):
        sys.exit(1)


def run_srt_cli(
    pipeline: KPipeline,
    language: str,
    voice: str,
    speed: float,
    verbose: bool,
    srt_file: str,
    output_file: str | None,
) -> None:
    """Generate timed audio from SRT file"""
    player = TTSPlayer(pipeline, language, voice, speed, verbose)

    if output_file is None:
        output_file = "srt_output.wav"

    try:
        console.print(f"[cyan]Processing SRT file:[/] {srt_file}")
        if not player.generate_srt_timed_audio(srt_file, output_file=output_file):
            sys.exit(1)
        console.print(f"[bold green]✓ Timed audio saved to:[/] {output_file}")
    except KeyboardInterrupt:
        console.print("[bold yellow]Exiting...[/]")
        sys.exit()
    except Exception as e:
        console.print(f"[bold red]Error processing SRT file:[/] {str(e)}")
        sys.exit(1)


def run_console(
    pipeline: KPipeline,
    language: str,
    voice: str,
    speed: float,
    verbose: bool,
    history_off: bool,
    device: str | None,
    ctrlc: bool,
    prompt="> ",
) -> None:
    """Run an interactive TTS session with dynamic settings."""

    player = TTSPlayer(pipeline, language, voice, speed, verbose, ctrlc)

    console.rule("[bold green]Interactive TTS started[/]")
    display_help()

    # Display starting configuration
    console.print("[green]Starting with:[/]")
    console.print(f"  Language: [cyan]{get_language_map()[language]}[/]")
    console.print(f"  Voice: [cyan]{voice}[/]")
    console.print(f"  Speed: [cyan]{speed}[/]")
    while True:
        try:
            user_input = get_input(history_off, prompt)
            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("!"):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == "!lang":
                    if player.change_language(arg, device):
                        console.print(
                            f"[green]Language changed to:[/] {player.languages[arg]}"
                        )
                    else:
                        console.print("[red]Invalid language code.[/]")
                        display_languages()

                elif cmd == "!voice":
                    if player.change_voice(arg):
                        console.print(f"[green]Voice changed to:[/] {arg}")
                    else:
                        console.print("[red]Invalid voice.[/]")
                        console.print("Use !list_voices to see options.")

                elif cmd == "!speed":
                    try:
                        new_speed = float(arg)
                        if player.change_speed(new_speed):
                            console.print(f"[green]Speed changed to:[/] {new_speed}")
                        else:
                            console.print(
                                f"[red]Speed must be between {MIN_SPEED} and {MAX_SPEED}[/]"
                            )
                    except ValueError:
                        console.print("[red]Invalid speed value[/]")

                elif cmd in ("!s", "!stop"):
                    player.stop_playback()

                elif cmd in ("!p", "!pause"):
                    player.pause_playback()

                elif cmd in ("!r", "!resume"):
                    player.resume_playback()

                elif cmd in ("!b", "!back"):
                    player.back_sentence()

                elif cmd in ("!n", "!next"):
                    player.skip_sentence()

                elif cmd == "!list_langs":
                    display_languages()

                elif cmd == "!list_voices":
                    display_voices(player.language)

                elif cmd == "!list_all_voices":
                    display_voices()

                elif cmd in ("!help", "!h"):
                    display_help()

                elif cmd in ("!quit", "!q"):
                    console.print("[bold yellow]Exiting...[/]")
                    stop_and_wait(player)
                    break

                elif cmd == "!clear":
                    print("\033[H\033[J", end="")

                elif cmd == "!clear_history":
                    clear_history()

                elif cmd == "!ctrlc":
                    player.ctrlc = not player.ctrlc
                    if player.ctrlc:
                        console.print("[green]Ctrl+C ends the playback")
                    else:
                        console.print("[green]Ctrl+C gives a new line")

                elif cmd == "!status":
                    display_status(player.language, player.voice, player.speed)

                elif cmd == "!verbose":
                    player.verbose = not player.verbose
                else:
                    console.print(f"[red]Unknown command: {cmd}[/]")
                    console.print("Type !help for available commands.")

                continue

            # Stop if previous playback still running
            stop_and_wait(player)

            sentences = split_text_to_sentences(user_input, player.nltk_language)
            with console.status(
                f"[cyan]Speaking:[/] {user_input[:30]}...", spinner_style="cyan"
            ):
                player.speak(sentences)

        except KeyboardInterrupt:
            if player.ctrlc:
                player.stop_playback(False)
                console.print("\n[bold yellow]Interrupted. Type !q to exit.[/]")
            else:
                player.print_complete = False
                console.print("\n[bold yellow]Type !p to pause.[/]")
        except EOFError:
            console.print("\n[bold yellow]Type !q to exit.[/]")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {str(e)}")
