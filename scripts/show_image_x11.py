#!/usr/bin/env python3
"""Show one image fullscreen on an X11 display for noVNC viewing."""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--title", default="ContactDiffusion Shadow grasp")
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    root = tk.Tk()
    root.title(args.title)
    root.configure(background="black")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # Lightweight X11 window managers may ignore the fullscreen hint. An
    # explicit borderless screen-sized geometry is deterministic under Xvfb.
    root.overrideredirect(True)
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.lift()
    root.focus_force()
    root.bind("<Escape>", lambda _event: root.destroy())
    root.bind("q", lambda _event: root.destroy())

    photo = tk.PhotoImage(file=str(image_path))
    canvas = tk.Canvas(root, background="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    def redraw(_event: object | None = None) -> None:
        canvas.delete("all")
        canvas.create_image(
            canvas.winfo_width() // 2,
            canvas.winfo_height() // 2,
            image=photo,
            anchor=tk.CENTER,
        )

    canvas.bind("<Configure>", redraw)
    root.after_idle(redraw)
    root.mainloop()


if __name__ == "__main__":
    main()
