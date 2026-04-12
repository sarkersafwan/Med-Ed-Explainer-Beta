"""Desktop GUI for the Medical Education Video Pipeline.

Simple tkinter interface to:
  - Upload a PDF or enter a topic
  - Configure duration, voice, avatar image
  - Run the pipeline with live progress output
  - View and open generated assets
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_AVATAR = PROJECT_ROOT / "MedVidSpeaker.png"
DEFAULT_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "")


class PipelineGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Medical Education Video Pipeline")
        self.root.geometry("820x720")
        self.root.configure(bg="#1e1e2e")

        self.process: subprocess.Popen | None = None
        self.pdf_path = tk.StringVar()
        self.topic_text = tk.StringVar()
        self.duration = tk.StringVar(value="0.33")
        self.voice_id = tk.StringVar(value=DEFAULT_VOICE_ID)
        self.avatar_image = tk.StringVar(
            value=str(DEFAULT_AVATAR) if DEFAULT_AVATAR.exists() else ""
        )
        self.input_mode = tk.StringVar(value="topic")

        # Skip flags
        self.skip_images = tk.BooleanVar(value=False)
        self.skip_voice = tk.BooleanVar(value=False)
        self.skip_avatar = tk.BooleanVar(value=False)
        self.skip_animation = tk.BooleanVar(value=False)
        self.dry_run = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("default")

        # Colors
        bg = "#1e1e2e"
        fg = "#cdd6f4"
        accent = "#89b4fa"
        surface = "#313244"
        green = "#a6e3a1"
        red = "#f38ba8"

        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Helvetica", 12))
        style.configure("Header.TLabel", background=bg, foreground=accent, font=("Helvetica", 16, "bold"))
        style.configure("TRadiobutton", background=bg, foreground=fg, font=("Helvetica", 12))
        style.configure("TCheckbutton", background=bg, foreground=fg, font=("Helvetica", 11))
        style.configure("Accent.TButton", background=accent, foreground="#1e1e2e", font=("Helvetica", 13, "bold"))
        style.configure("TEntry", fieldbackground=surface, foreground=fg)

        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)

        # --- Header ---
        ttk.Label(main, text="Medical Education Video Pipeline", style="Header.TLabel").pack(anchor="w")
        ttk.Label(main, text="Generate AI explainer videos from PDFs or topics", foreground="#6c7086").pack(anchor="w", pady=(0, 15))

        # --- Input Source ---
        source_frame = ttk.Frame(main)
        source_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(source_frame, text="Input Source:").pack(anchor="w")

        radio_frame = ttk.Frame(source_frame)
        radio_frame.pack(fill="x", pady=5)
        ttk.Radiobutton(radio_frame, text="Topic", variable=self.input_mode, value="topic", command=self._toggle_input).pack(side="left", padx=(0, 20))
        ttk.Radiobutton(radio_frame, text="PDF File", variable=self.input_mode, value="pdf", command=self._toggle_input).pack(side="left")

        # Topic input
        self.topic_frame = ttk.Frame(source_frame)
        self.topic_frame.pack(fill="x", pady=5)
        self.topic_entry = tk.Entry(self.topic_frame, textvariable=self.topic_text, font=("Helvetica", 12), bg=surface, fg=fg, insertbackground=fg, relief="flat", bd=8)
        self.topic_entry.pack(fill="x")
        self.topic_entry.insert(0, "e.g. Muscle Contraction")
        self.topic_entry.bind("<FocusIn>", self._clear_placeholder)
        self.topic_entry.bind("<FocusOut>", self._restore_placeholder)

        # PDF input
        self.pdf_frame = ttk.Frame(source_frame)
        pdf_row = ttk.Frame(self.pdf_frame)
        pdf_row.pack(fill="x")
        tk.Entry(pdf_row, textvariable=self.pdf_path, font=("Helvetica", 12), bg=surface, fg=fg, insertbackground=fg, relief="flat", bd=8).pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(pdf_row, text="Browse", command=self._browse_pdf, bg=accent, fg="#1e1e2e", font=("Helvetica", 11, "bold"), relief="flat", padx=12, pady=4).pack(side="right")

        # --- Settings Row ---
        settings = ttk.Frame(main)
        settings.pack(fill="x", pady=10)

        # Duration
        dur_frame = ttk.Frame(settings)
        dur_frame.pack(side="left", padx=(0, 20))
        ttk.Label(dur_frame, text="Duration (min):").pack(anchor="w")
        tk.Entry(dur_frame, textvariable=self.duration, width=8, font=("Helvetica", 12), bg=surface, fg=fg, insertbackground=fg, relief="flat", bd=6).pack(anchor="w")

        # Voice ID
        voice_frame = ttk.Frame(settings)
        voice_frame.pack(side="left", padx=(0, 20), fill="x", expand=True)
        ttk.Label(voice_frame, text="Voice ID:").pack(anchor="w")
        tk.Entry(voice_frame, textvariable=self.voice_id, font=("Helvetica", 12), bg=surface, fg=fg, insertbackground=fg, relief="flat", bd=6).pack(fill="x")

        # Avatar image
        avatar_frame = ttk.Frame(main)
        avatar_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(avatar_frame, text="Avatar Image:").pack(anchor="w")
        avatar_row = ttk.Frame(avatar_frame)
        avatar_row.pack(fill="x")
        tk.Entry(avatar_row, textvariable=self.avatar_image, font=("Helvetica", 11), bg=surface, fg=fg, insertbackground=fg, relief="flat", bd=6).pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(avatar_row, text="Browse", command=self._browse_avatar, bg=accent, fg="#1e1e2e", font=("Helvetica", 11, "bold"), relief="flat", padx=12, pady=4).pack(side="right")

        # --- Skip Options ---
        skip_frame = ttk.Frame(main)
        skip_frame.pack(fill="x", pady=(0, 10))
        ttk.Checkbutton(skip_frame, text="Skip Images", variable=self.skip_images).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(skip_frame, text="Skip Voice", variable=self.skip_voice).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(skip_frame, text="Skip Avatar", variable=self.skip_avatar).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(skip_frame, text="Skip Animation", variable=self.skip_animation).pack(side="left", padx=(0, 15))
        ttk.Checkbutton(skip_frame, text="Dry Run", variable=self.dry_run).pack(side="left")

        # --- Run / Stop Buttons ---
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill="x", pady=(0, 10))

        self.run_btn = tk.Button(
            btn_frame, text="Run Pipeline", command=self._run_pipeline,
            bg=green, fg="#1e1e2e", font=("Helvetica", 14, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2",
        )
        self.run_btn.pack(side="left", padx=(0, 10))

        self.stop_btn = tk.Button(
            btn_frame, text="Stop", command=self._stop_pipeline,
            bg=red, fg="#1e1e2e", font=("Helvetica", 14, "bold"),
            relief="flat", padx=20, pady=8, cursor="hand2", state="disabled",
        )
        self.stop_btn.pack(side="left", padx=(0, 10))

        self.open_btn = tk.Button(
            btn_frame, text="Open Output Folder", command=self._open_output,
            bg=surface, fg=fg, font=("Helvetica", 12),
            relief="flat", padx=16, pady=8, cursor="hand2",
        )
        self.open_btn.pack(side="right")

        # --- Log Output ---
        ttk.Label(main, text="Pipeline Output:").pack(anchor="w", pady=(5, 2))
        self.log = scrolledtext.ScrolledText(
            main, height=15, font=("Menlo", 11), bg="#11111b", fg=fg,
            insertbackground=fg, relief="flat", bd=8, wrap="word",
        )
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var, bg="#11111b", fg="#6c7086",
            font=("Helvetica", 10), anchor="w", padx=10, pady=4,
        )
        status_bar.pack(fill="x", side="bottom")

    def _toggle_input(self) -> None:
        if self.input_mode.get() == "topic":
            self.pdf_frame.pack_forget()
            self.topic_frame.pack(fill="x", pady=5)
        else:
            self.topic_frame.pack_forget()
            self.pdf_frame.pack(fill="x", pady=5)

    def _clear_placeholder(self, event: tk.Event) -> None:
        if self.topic_entry.get() == "e.g. Muscle Contraction":
            self.topic_entry.delete(0, "end")

    def _restore_placeholder(self, event: tk.Event) -> None:
        if not self.topic_entry.get().strip():
            self.topic_entry.insert(0, "e.g. Muscle Contraction")

    def _browse_pdf(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Medical Education PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
        )
        if path:
            self.pdf_path.set(path)

    def _browse_avatar(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Avatar Reference Image",
            filetypes=[("Images", "*.png *.jpg *.jpeg"), ("All files", "*.*")],
        )
        if path:
            self.avatar_image.set(path)

    def _log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _run_pipeline(self) -> None:
        # Build command
        cmd = [sys.executable, "run.py"]

        if self.input_mode.get() == "pdf":
            pdf = self.pdf_path.get().strip()
            if not pdf:
                messagebox.showerror("Error", "Please select a PDF file.")
                return
            cmd.append(pdf)
        else:
            topic = self.topic_text.get().strip()
            if not topic or topic == "e.g. Muscle Contraction":
                messagebox.showerror("Error", "Please enter a topic.")
                return
            cmd.extend(["--topic", topic])

        # Duration
        dur = self.duration.get().strip()
        if dur:
            cmd.extend(["--duration", dur])

        # Voice
        vid = self.voice_id.get().strip()
        if vid:
            cmd.extend(["--voice-id", vid])

        # Avatar
        avatar = self.avatar_image.get().strip()
        if avatar:
            cmd.extend(["--avatar-image", avatar])

        # Skips
        if self.skip_images.get():
            cmd.append("--skip-images")
        if self.skip_voice.get():
            cmd.append("--skip-voice")
        if self.skip_avatar.get():
            cmd.append("--skip-avatar")
        if self.skip_animation.get():
            cmd.append("--skip-animation")
        if self.dry_run.get():
            cmd.append("--dry-run")

        self._clear_log()
        self._log(f"$ {' '.join(cmd)}\n\n")
        self.status_var.set("Running pipeline...")
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")

        # Run in background thread
        thread = threading.Thread(target=self._execute_pipeline, args=(cmd,), daemon=True)
        thread.start()

    def _execute_pipeline(self, cmd: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
            )

            for line in self.process.stdout:
                self.root.after(0, self._log, line)

            self.process.wait()
            exit_code = self.process.returncode

            if exit_code == 0:
                self.root.after(0, self._log, "\n--- Pipeline completed successfully! ---\n")
                self.root.after(0, self.status_var.set, "Done!")
            else:
                self.root.after(0, self._log, f"\n--- Pipeline exited with code {exit_code} ---\n")
                self.root.after(0, self.status_var.set, f"Failed (exit code {exit_code})")

        except Exception as e:
            self.root.after(0, self._log, f"\nError: {e}\n")
            self.root.after(0, self.status_var.set, "Error")
        finally:
            self.process = None
            self.root.after(0, self.run_btn.configure, {"state": "normal"})
            self.root.after(0, self.stop_btn.configure, {"state": "disabled"})

    def _stop_pipeline(self) -> None:
        if self.process:
            self.process.terminate()
            self._log("\n--- Pipeline stopped by user ---\n")
            self.status_var.set("Stopped")

    def _open_output(self) -> None:
        # Try to find the most recent output folder
        if OUTPUT_DIR.exists():
            folders = sorted(OUTPUT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            target = folders[0] if folders else OUTPUT_DIR
        else:
            target = PROJECT_ROOT

        if sys.platform == "darwin":
            subprocess.run(["open", str(target)])
        elif sys.platform == "win32":
            subprocess.run(["explorer", str(target)])
        else:
            subprocess.run(["xdg-open", str(target)])


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
