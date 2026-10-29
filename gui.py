"""
PCB Component Detector - Live Demo GUI
Run this file to launch the detector.
Keys: Q = quit, S = screenshot, +/- = confidence
"""
import tkinter as tk
from tkinter import font as tkfont
import threading
import time
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

ROOT     = Path(__file__).resolve().parent
WEIGHTS  = ROOT / "runs" / "pcb" / "weights" / "best.pt"
CAM_IDX  = 0
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


class PCBDetectorGUI:
    def __init__(self, root):
        self.root    = root
        self.root.title(TITLE)
        self.root.configure(bg=BG)
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)

        self.model   = YOLO(str(WEIGHTS))
        self.names   = self.model.names
        self.colors  = Colors()
        self.conf    = tk.DoubleVar(value=0.25)
        self.running = False
        self.frozen  = False
        self.frame   = None
        self.cap     = None
        self.shot_n  = 0

        self._build_ui()
        self._start_camera()

    # ── UI BUILD ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self.root, bg=BG, pady=14)
        hdr.pack(fill="x", padx=24)

        title_font = tkfont.Font(family="Helvetica Neue", size=22, weight="bold")
        sub_font   = tkfont.Font(family="Helvetica Neue", size=11)

        tk.Label(hdr, text=TITLE,    font=title_font, bg=BG, fg=TEXT   ).pack(side="left")
        tk.Label(hdr, text=SUBTITLE, font=sub_font,   bg=BG, fg=SUBTEXT).pack(side="left", padx=(12, 0), pady=(6, 0))
        tk.Label(hdr, text=TEAM,     font=sub_font,   bg=BG, fg=ACCENT ).pack(side="right")

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=24)

        # Main body: camera + sidebar
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=16)

        # Camera feed (left)
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        cam_wrap = tk.Frame(left, bg=PANEL_BG, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
        cam_wrap.pack(fill="both", expand=True)

        self.cam_label = tk.Label(cam_wrap, bg=PANEL_BG,
                                  text="Starting camera…",
                                  fg=SUBTEXT, font=("Helvetica Neue", 13))
        self.cam_label.pack(fill="both", expand=True)

        # Sidebar (right)
        self.sidebar = tk.Frame(body, bg=BG, width=260)
        self.sidebar.pack(side="right", fill="y", padx=(16, 0))
        self.sidebar.pack_propagate(False)
        self._build_sidebar()

        # Bottom legend bar
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=24)
        legend_bar = tk.Frame(self.root, bg=PANEL_BG, pady=10)
        legend_bar.pack(fill="x", padx=24, pady=(0, 10))

        tk.Label(legend_bar, text="LEGEND", fg=SUBTEXT, bg=PANEL_BG,
                 font=("Helvetica Neue", 9, "bold")).pack(side="left", padx=(14, 20))

        for i, name in self.names.items():
            rgb  = self.colors(i, bgr=False)
            hex_ = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            item = tk.Frame(legend_bar, bg=PANEL_BG)
            item.pack(side="left", padx=14)
            tk.Label(item, bg=hex_, width=3, height=1,
                     relief="flat").pack(side="left", padx=(0, 6))
            tk.Label(item, text=name, fg=TEXT, bg=PANEL_BG,
                     font=("Helvetica Neue", 10)).pack(side="left")

    def _build_sidebar(self):
        sb = self.sidebar

        # Status
        self._section(sb, "Status")
        status_card = tk.Frame(sb, bg=CARD_BG, bd=0,
                               highlightthickness=1, highlightbackground=BORDER)
        status_card.pack(fill="x")
        self.status_dot = tk.Label(status_card, text="●", fg=GREEN,
                                   bg=CARD_BG, font=("Helvetica Neue", 13))
        self.status_dot.pack(anchor="w", padx=14, pady=(8, 0))
        self.status_lbl = tk.Label(status_card, text="Camera active",
                                   fg=TEXT, bg=CARD_BG,
                                   font=("Helvetica Neue", 11))
        self.status_lbl.pack(anchor="w", padx=14)
        self.fps_lbl = tk.Label(status_card, text="FPS: —",
                                fg=SUBTEXT, bg=CARD_BG,
                                font=("Helvetica Neue", 10))
        self.fps_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        # Confidence
        self._section(sb, "Confidence Threshold")
        conf_card = tk.Frame(sb, bg=CARD_BG, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        conf_card.pack(fill="x")
        self.conf_val_lbl = tk.Label(conf_card,
                                     text=f"{self.conf.get():.2f}",
                                     fg=ACCENT, bg=CARD_BG,
                                     font=("Helvetica Neue", 18, "bold"))
        self.conf_val_lbl.pack(pady=(8, 4))
        btn_row = tk.Frame(conf_card, bg=CARD_BG)
        btn_row.pack(fill="x", padx=14, pady=(0, 10))
        self._btn(btn_row, "−", self._conf_down).pack(side="left", expand=True, fill="x", padx=(0, 4))
        self._btn(btn_row, "+", self._conf_up  ).pack(side="left", expand=True, fill="x")

        # Detections
        self._section(sb, "Detections")
        det_card = tk.Frame(sb, bg=CARD_BG, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
        det_card.pack(fill="x")
        self.detect_frame = tk.Frame(det_card, bg=CARD_BG)
        self.detect_frame.pack(fill="x", padx=14, pady=(8, 4))
        self.total_lbl = tk.Label(det_card, text="Total components: 0",
                                  fg=SUBTEXT, bg=CARD_BG,
                                  font=("Helvetica Neue", 10))
        self.total_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        # Controls
        self._section(sb, "Controls")
        ctrl_card = tk.Frame(sb, bg=CARD_BG, bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        ctrl_card.pack(fill="x")
        ctrl = tk.Frame(ctrl_card, bg=CARD_BG)
        ctrl.pack(fill="x", padx=14, pady=10)
        self.freeze_btn = self._btn(ctrl, "⏸  Freeze", self._toggle_freeze, color=ACCENT2)
        self.freeze_btn.pack(fill="x", pady=(0, 6))
        self._btn(ctrl, "📷  Screenshot", self._screenshot, color="#6E40C9").pack(fill="x")

        tk.Label(sb, text="Press Q to quit", fg=SUBTEXT,
                 bg=BG, font=("Helvetica Neue", 9)).pack(side="bottom", pady=8)

    def _section(self, parent, title):
        tk.Label(parent, text=title.upper(), fg=SUBTEXT, bg=BG,
                 font=("Helvetica Neue", 9, "bold"),
                 anchor="w").pack(fill="x", pady=(14, 3))

    def _btn(self, parent, text, cmd, color=BORDER):
        return tk.Button(parent, text=text, command=cmd,
                         bg=color, fg=TEXT, relief="flat",
                         font=("Helvetica Neue", 11),
                         activebackground=ACCENT, activeforeground=BG,
                         cursor="hand2", padx=8, pady=6)

    # ── CAMERA LOOP ───────────────────────────────────────────────────────────

    def _start_camera(self):
        self.cap = cv2.VideoCapture(CAM_IDX)
        if not self.cap.isOpened():
            self.status_lbl.config(text="Camera not found", fg=RED)
            self.status_dot.config(fg=RED)
            return
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        self.root.bind("<q>", lambda e: self._quit())
        self.root.bind("<Q>", lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _loop(self):
        prev = time.time()
        while self.running:
            if not self.frozen:
                ok, raw = self.cap.read()
                if not ok:
                    break
                self.frame = raw

            if self.frame is None:
                time.sleep(0.03)
                continue

            result    = self.model.predict(self.frame, conf=self.conf.get(),
                                           verbose=False, hide_conf=True)[0]
            annotated = result.plot(line_width=1, font_size=0.4)
            counts    = Counter(self.names[int(c)] for c in result.boxes.cls)

            now  = time.time()
            fps  = 1 / max(now - prev, 1e-6)
            prev = now

            self.root.after(0, self._update_ui, annotated, counts, fps)
            time.sleep(0.01)

    def _update_ui(self, frame_bgr, counts, fps):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        w = self.cam_label.winfo_width()
        h = self.cam_label.winfo_height()
        if w > 10 and h > 10:
            pil = pil.resize((w, h), Image.LANCZOS)

        imgtk = ImageTk.PhotoImage(pil)
        self.cam_label.config(image=imgtk, text="")
        self.cam_label.image = imgtk

        self.fps_lbl.config(text=f"FPS: {fps:.1f}")

        for widget in self.detect_frame.winfo_children():
            widget.destroy()

        if counts:
            for name, n in sorted(counts.items()):
                i    = list(self.names.values()).index(name)
                rgb2 = self.colors(i, bgr=False)
                hex_ = "#{:02x}{:02x}{:02x}".format(int(rgb2[0]), int(rgb2[1]), int(rgb2[2]))
                row  = tk.Frame(self.detect_frame, bg=CARD_BG)
                row.pack(fill="x", pady=1)
                tk.Label(row, bg=hex_, width=2, relief="flat").pack(side="left", padx=(0, 6))
                tk.Label(row, text=name, fg=TEXT, bg=CARD_BG,
                         font=("Helvetica Neue", 10),
                         anchor="w").pack(side="left", expand=True, fill="x")
                tk.Label(row, text=str(n), fg=ACCENT, bg=CARD_BG,
                         font=("Helvetica Neue", 10, "bold")).pack(side="right")
        else:
            tk.Label(self.detect_frame, text="Nothing detected",
                     fg=SUBTEXT, bg=CARD_BG,
                     font=("Helvetica Neue", 10)).pack(pady=4)

        self.total_lbl.config(text=f"Total components: {sum(counts.values())}")

    # ── CONTROLS ──────────────────────────────────────────────────────────────

    def _conf_up(self):
        v = min(0.95, self.conf.get() + 0.05)
        self.conf.set(round(v, 2))
        self.conf_val_lbl.config(text=f"{v:.2f}")

    def _conf_down(self):
        v = max(0.05, self.conf.get() - 0.05)
        self.conf.set(round(v, 2))
        self.conf_val_lbl.config(text=f"{v:.2f}")

    def _toggle_freeze(self):
        self.frozen = not self.frozen
        if self.frozen:
            self.freeze_btn.config(text="▶  Resume", bg=GREEN)
            self.status_lbl.config(text="Frozen")
            self.status_dot.config(fg="#E3B341")
        else:
            self.freeze_btn.config(text="⏸  Freeze", bg=ACCENT2)
            self.status_lbl.config(text="Camera active")
            self.status_dot.config(fg=GREEN)

    def _screenshot(self):
        if self.frame is None:
            return
        SHOT_DIR.mkdir(exist_ok=True)
        self.shot_n += 1
        path = SHOT_DIR / f"screenshot_{self.shot_n}.jpg"
        cv2.imwrite(str(path), self.frame)
        self.status_lbl.config(text=f"Saved screenshot {self.shot_n}")
        self.root.after(2000, lambda: self.status_lbl.config(text="Camera active"))

    def _quit(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app  = PCBDetectorGUI(root)
    root.mainloop()