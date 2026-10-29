"""
PCB Component Identifier - Live Demo GUI
Keys: Q = quit, S = screenshot, F = freeze, + / - = confidence
"""
import threading
import time
import tkinter as tk
from collections import Counter
from pathlib import Path
from tkinter import font as tkfont

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

ROOT     = Path(__file__).resolve().parent
WEIGHTS  = ROOT / "runs" / "pcb" / "weights" / "best.pt"
CAM_IDX  = 0                      # try 1 if the wrong camera opens
IMGSZ    = 640                    # lower to 480 for a faster (but less accurate) feed
SHOT_DIR = ROOT / "screenshots"

TITLE    = "PCB Component Identifier"
SUBTITLE = "Real-Time Detection using YOLOv11s"

BG       = "#0D1117"
PANEL_BG = "#161B22"
ACCENT   = "#00C6FF"
ACCENT2  = "#0072FF"
TEXT     = "#E6EDF3"
SUBTEXT  = "#8B949E"
CARD_BG  = "#21262D"
BORDER   = "#30363D"
GREEN    = "#3FB950"
RED      = "#F85149"
AMBER    = "#E3B341"
PURPLE   = "#6E40C9"

FONT = "Segoe UI"


def to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


class PCBDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(TITLE)
        self.root.configure(bg=BG)

        # Size the window to fit THIS screen (fixes cut-off sidebar / legend)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1280, sw - 40), min(820, sh - 120)
        self.root.geometry(f"{w}x{h}+20+10")
        self.root.minsize(900, 560)

        self.model  = YOLO(str(WEIGHTS))
        self.names  = self.model.names
        self.colors = Colors()
        self.conf   = tk.DoubleVar(value=0.25)

        self.running        = False
        self.frozen         = False
        self.frame          = None      # latest raw camera frame
        self.last_annotated = None      # latest frame with boxes drawn
        self.last_conf      = None
        self.cap            = None
        self.shot_n         = 0

        self._build_ui()
        self._start_camera()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, pady=8)
        hdr.pack(fill="x", padx=20)
        title_font = tkfont.Font(family=FONT, size=20, weight="bold")
        sub_font   = tkfont.Font(family=FONT, size=11)
        tk.Label(hdr, text=TITLE, font=title_font, bg=BG, fg=TEXT).pack(side="left")
        tk.Label(hdr, text=SUBTITLE, font=sub_font, bg=BG, fg=SUBTEXT).pack(
            side="left", padx=(12, 0), pady=(6, 0))
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=20)

        # Legend bar - packed at the bottom FIRST so it is never pushed off screen
        legend_bar = tk.Frame(self.root, bg=PANEL_BG, pady=8)
        legend_bar.pack(side="bottom", fill="x", padx=20, pady=(0, 8))
        tk.Frame(self.root, bg=BORDER, height=1).pack(side="bottom", fill="x", padx=20)

        tk.Label(legend_bar, text="Legend", fg=SUBTEXT, bg=PANEL_BG,
                 font=(FONT, 10, "bold")).pack(side="left", padx=(14, 18))
        for i, name in self.names.items():
            item = tk.Frame(legend_bar, bg=PANEL_BG)
            item.pack(side="left", padx=12)
            tk.Label(item, bg=to_hex(self.colors(i, bgr=False)),
                     width=3, height=1).pack(side="left", padx=(0, 6))
            tk.Label(item, text=name, fg=TEXT, bg=PANEL_BG,
                     font=(FONT, 10)).pack(side="left")

        # Body. The sidebar is packed FIRST so it always keeps its full width;
        # the camera area takes whatever space is left.
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=10)

        self.sidebar = tk.Frame(body, bg=BG, width=300)
        self.sidebar.pack(side="right", fill="y", padx=(12, 0))
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        self.cam_wrap = tk.Frame(body, bg=PANEL_BG, width=100, height=100,
                                 highlightthickness=1, highlightbackground=BORDER)
        self.cam_wrap.pack(side="left", fill="both", expand=True)
        self.cam_wrap.pack_propagate(False)   # the image can never resize its own frame

        self.cam_label = tk.Label(self.cam_wrap, bg=PANEL_BG, text="Starting camera...",
                                  fg=SUBTEXT, font=(FONT, 13))
        self.cam_label.pack(expand=True)

    def _build_sidebar(self):
        sb = self.sidebar

        # Status
        self._section(sb, "Status")
        card = self._card(sb)
        row = tk.Frame(card, bg=CARD_BG)
        row.pack(fill="x", padx=12, pady=8)
        self.status_dot = tk.Label(row, text="●", fg=GREEN, bg=CARD_BG, font=(FONT, 12))
        self.status_dot.pack(side="left")
        self.status_lbl = tk.Label(row, text="Camera active", fg=TEXT,
                                   bg=CARD_BG, font=(FONT, 11))
        self.status_lbl.pack(side="left", padx=(6, 0))
        self.fps_lbl = tk.Label(row, text="FPS: -", fg=SUBTEXT, bg=CARD_BG, font=(FONT, 10))
        self.fps_lbl.pack(side="right")

        # Confidence
        self._section(sb, "Confidence threshold")
        card = self._card(sb)
        row = tk.Frame(card, bg=CARD_BG)
        row.pack(fill="x", padx=12, pady=8)
        self._btn(row, "−", self._conf_down, width=4).pack(side="left")
        self.conf_val_lbl = tk.Label(row, text=f"{self.conf.get():.2f}", fg=ACCENT,
                                     bg=CARD_BG, font=(FONT, 16, "bold"))
        self.conf_val_lbl.pack(side="left", expand=True)
        self._btn(row, "+", self._conf_up, width=4).pack(side="right")

        # Detections
        self._section(sb, "Detections")
        card = self._card(sb)
        self.detect_frame = tk.Frame(card, bg=CARD_BG)
        self.detect_frame.pack(fill="x", padx=12, pady=(8, 2))
        self.total_lbl = tk.Label(card, text="Total components: 0", fg=SUBTEXT,
                                  bg=CARD_BG, font=(FONT, 10))
        self.total_lbl.pack(anchor="w", padx=12, pady=(0, 8))

        # Controls
        self._section(sb, "Controls")
        card = self._card(sb)
        ctrl = tk.Frame(card, bg=CARD_BG)
        ctrl.pack(fill="x", padx=12, pady=10)
        self.freeze_btn = self._btn(ctrl, "Freeze", self._toggle_freeze, color=ACCENT2)
        self.freeze_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn(ctrl, "Screenshot", self._screenshot, color=PURPLE).pack(
            side="left", expand=True, fill="x")

        tk.Label(sb, text="Keys: F freeze, S screenshot, Q quit", fg=SUBTEXT,
                 bg=BG, font=(FONT, 9)).pack(side="bottom", pady=4)

    def _section(self, parent, title):
        tk.Label(parent, text=title, fg=SUBTEXT, bg=BG, font=(FONT, 10, "bold"),
                 anchor="w").pack(fill="x", pady=(10, 3))

    def _card(self, parent):
        f = tk.Frame(parent, bg=CARD_BG, highlightthickness=1, highlightbackground=BORDER)
        f.pack(fill="x")
        return f

    def _btn(self, parent, text, cmd, color=BORDER, width=None):
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg=TEXT, relief="flat",
                      font=(FONT, 11), activebackground=ACCENT, activeforeground=BG,
                      cursor="hand2", padx=8, pady=5)
        if width:
            b.config(width=width)
        return b

    # -------------------------------------------------------------- CAMERA

    def _start_camera(self):
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.bind("<f>", lambda e: self._toggle_freeze())
        self.root.bind("<s>", lambda e: self._screenshot())
        self.root.bind("<plus>", lambda e: self._conf_up())
        self.root.bind("<equal>", lambda e: self._conf_up())
        self.root.bind("<minus>", lambda e: self._conf_down())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

        self.cap = cv2.VideoCapture(CAM_IDX)
        if not self.cap.isOpened():
            self.cam_label.config(text="Camera not found.\nCheck the camera or change CAM_IDX.")
            self.status_lbl.config(text="Camera not found", fg=RED)
            self.status_dot.config(fg=RED)
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = time.time()
        while self.running:
            conf = round(self.conf.get(), 2)

            if self.frozen and self.last_annotated is not None and conf == self.last_conf:
                time.sleep(0.05)       # nothing new to compute while frozen
                continue

            if not self.frozen:
                ok, raw = self.cap.read()
                if not ok:
                    break
                self.frame = raw

            if self.frame is None:
                time.sleep(0.03)
                continue

            result = self.model.predict(self.frame, conf=conf, imgsz=IMGSZ,
                                        verbose=False, max_det=1000)[0]
            # Boxes only: no class text, no confidence numbers
            annotated = result.plot(labels=False, conf=False, line_width=1)
            counts    = Counter(self.names[int(c)] for c in result.boxes.cls)

            self.last_annotated = annotated
            self.last_conf      = conf

            now  = time.time()
            fps  = 1 / max(now - prev, 1e-6)
            prev = now

            try:
                self.root.after(0, self._update_ui, annotated, counts, fps)
            except (tk.TclError, RuntimeError):
                break
            time.sleep(0.01)

    def _update_ui(self, frame_bgr, counts, fps):
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        # Fit the image inside the camera area, keeping the aspect ratio
        w = self.cam_wrap.winfo_width() - 4
        h = self.cam_wrap.winfo_height() - 4
        if w > 10 and h > 10:
            scale = min(w / pil.width, h / pil.height)
            pil = pil.resize((max(1, int(pil.width * scale)),
                              max(1, int(pil.height * scale))), Image.BILINEAR)

        imgtk = ImageTk.PhotoImage(pil)
        self.cam_label.config(image=imgtk, text="")
        self.cam_label.image = imgtk

        if not self.frozen:
            self.fps_lbl.config(text=f"FPS: {fps:.1f}")

        for widget in self.detect_frame.winfo_children():
            widget.destroy()

        if counts:
            names_list = list(self.names.values())
            for name, n in sorted(counts.items()):
                color = to_hex(self.colors(names_list.index(name), bgr=False))
                row = tk.Frame(self.detect_frame, bg=CARD_BG)
                row.pack(fill="x", pady=1)
                tk.Label(row, bg=color, width=2).pack(side="left", padx=(0, 8))
                tk.Label(row, text=name, fg=TEXT, bg=CARD_BG, font=(FONT, 10),
                         anchor="w").pack(side="left", expand=True, fill="x")
                tk.Label(row, text=str(n), fg=ACCENT, bg=CARD_BG,
                         font=(FONT, 10, "bold")).pack(side="right")
        else:
            tk.Label(self.detect_frame, text="Nothing detected", fg=SUBTEXT,
                     bg=CARD_BG, font=(FONT, 10)).pack(pady=4)

        self.total_lbl.config(text=f"Total components: {sum(counts.values())}")

    # ------------------------------------------------------------ CONTROLS

    def _conf_up(self):
        v = round(min(0.95, self.conf.get() + 0.05), 2)
        self.conf.set(v)
        self.conf_val_lbl.config(text=f"{v:.2f}")

    def _conf_down(self):
        v = round(max(0.05, self.conf.get() - 0.05), 2)
        self.conf.set(v)
        self.conf_val_lbl.config(text=f"{v:.2f}")

    def _toggle_freeze(self):
        self.frozen = not self.frozen
        if self.frozen:
            self.freeze_btn.config(text="Resume", bg=GREEN)
            self.status_lbl.config(text="Frozen")
            self.status_dot.config(fg=AMBER)
        else:
            self.freeze_btn.config(text="Freeze", bg=ACCENT2)
            self.status_lbl.config(text="Camera active")
            self.status_dot.config(fg=GREEN)

    def _screenshot(self):
        if self.last_annotated is None:
            return
        SHOT_DIR.mkdir(exist_ok=True)
        self.shot_n += 1
        path = SHOT_DIR / f"screenshot_{self.shot_n}.jpg"
        cv2.imwrite(str(path), self.last_annotated)       # saves WITH the boxes
        self.status_lbl.config(text=f"Saved screenshot {self.shot_n}")
        self.root.after(2000, lambda: self.status_lbl.config(
            text="Frozen" if self.frozen else "Camera active"))

    def _quit(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PCBDetectorGUI(root)
    root.mainloop()