# -*- coding: utf-8 -*-
"""
NHAN DANG CHU CAI VIET TAY A-Z  (v6.0 – HUD/Sci-Fi Edition)
Dense MLP · Keras/TensorFlow · Online Learning · Tkinter
pip install tensorflow pillow numpy
python app.py
"""
import tkinter as tk
from tkinter import messagebox
import numpy as np
import os, sys, time
from datetime import datetime

try:
    from PIL import Image, ImageDraw
except ImportError:
    print("[LOI] pip install pillow"); sys.exit(1)
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import to_categorical
except ImportError:
    print("[LOI] pip install tensorflow"); sys.exit(1)

# ══════════════════════════════════════════════════════════════════
# CONFIG & CONSTANTS
# ══════════════════════════════════════════════════════════════════
MODEL_PATH  = "chucai.keras"
SAVE_DIR    = "du_lieu_thu_thap"
CANVAS_SZ   = 390
IMG_SIZE    = 28
NHAN        = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_HIST    = 12
DEFAULT_PEN = 14

# HUD Color Palette
BG          = "#03070F"   # App background (near black)
GRID_LINE   = "#0A1628"   # Grid lines on background
PANEL_BG    = "#060D1A"   # Panel background
PANEL_DARK  = "#040810"   # Darker panel
CARD_BG     = "#0A1628"   # Card inner
CARD_BG2    = "#0D1E36"   # Alt card

CYAN        = "#00E5FF"   # Electric cyan – primary accent
CYAN2       = "#00B8D4"   # Dimmer cyan
CYAN_DARK   = "#003D4D"   # Dark cyan for fills
CYAN_GLOW   = "#004A5E"   # Glow bg
TEAL        = "#00E676"   # Green confirm
TEAL_DARK   = "#003B1F"
RED_C       = "#FF1744"   # Red reject
RED_DARK    = "#3D0010"
AMBER       = "#FFD740"   # Warning / correction
AMBER_DARK  = "#3D3000"
PURPLE      = "#D500F9"   # Learn accent
PURPLE_DARK = "#2A003D"

WHITE       = "#E8F4FD"
MUTED       = "#3A5A7A"
DIM         = "#152030"

# Canvas drawing colors
CV_BG       = "#040D1A"   # Dark canvas
CV_INK      = "#FFFFFF"   # White ink (visual)
PIL_BG      = "#FFFFFF"   # PIL image – white bg
PIL_INK     = "#000000"   # PIL image – black ink

# Fonts
F_TITLE     = ("Courier New", 15, "bold")
F_SUBTITLE  = ("Courier New", 10, "bold")
F_HEAD      = ("Courier New", 9, "bold")
F_BODY      = ("Courier New", 9)
F_TINY      = ("Courier New", 8)
F_MONO      = ("Courier New", 9)
F_CHAR      = ("Courier New", 90, "bold")
F_PCT       = ("Courier New", 30, "bold")
F_BTN       = ("Courier New", 10, "bold")


# ══════════════════════════════════════════════════════════════════
# HUD HELPER: draw corner brackets on a Canvas
# ══════════════════════════════════════════════════════════════════
def draw_hud_corners(cv, x1, y1, x2, y2, size=18, color=CYAN, width=2):
    """Draw 4 corner L-brackets inside a Canvas at given coords."""
    s = size
    kw = dict(fill=color, width=width)
    # Top-left
    cv.create_line(x1, y1+s, x1, y1, x1+s, y1, **kw)
    # Top-right
    cv.create_line(x2-s, y1, x2, y1, x2, y1+s, **kw)
    # Bottom-left
    cv.create_line(x1, y2-s, x1, y2, x1+s, y2, **kw)
    # Bottom-right
    cv.create_line(x2-s, y2, x2, y2, x2, y2-s, **kw)


def make_hud_frame(parent, bg=PANEL_BG, border_color=CYAN,
                   padx=0, pady=0, corner=14):
    """Return (outer_canvas, inner_frame) that looks like a HUD panel."""
    outer = tk.Frame(parent, bg=border_color, padx=1, pady=1)
    inner = tk.Frame(outer, bg=bg, padx=padx, pady=pady)
    inner.pack(fill="both", expand=True)
    return outer, inner


# ══════════════════════════════════════════════════════════════════
# SCROLLABLE FRAME
# ══════════════════════════════════════════════════════════════════
class ScrollFrame(tk.Frame):
    def __init__(self, parent, bg=PANEL_BG, **kw):
        super().__init__(parent, bg=bg, **kw)
        vs = tk.Scrollbar(self, orient="vertical", bg=PANEL_DARK,
                          troughcolor=PANEL_DARK, width=5,
                          relief="flat", bd=0)
        vs.pack(side="right", fill="y")
        self._cv = tk.Canvas(self, bg=bg, highlightthickness=0,
                             yscrollcommand=vs.set)
        self._cv.pack(side="left", fill="both", expand=True)
        vs.config(command=self._cv.yview)
        self.inner = tk.Frame(self._cv, bg=bg)
        self._wid = self._cv.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>",
                        lambda e: self._cv.configure(
                            scrollregion=self._cv.bbox("all")))
        self._cv.bind("<Configure>",
                      lambda e: self._cv.itemconfig(self._wid, width=e.width))
        for w in (self._cv, self.inner):
            w.bind("<MouseWheel>",
                   lambda e: self._cv.yview_scroll(int(-1*(e.delta/120)), "units"))

    def scroll_top(self): self._cv.yview_moveto(0)
    def scroll_bot(self): self._cv.yview_moveto(1)


# ══════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════
class App:
    def __init__(self, root: tk.Tk):
        self.root          = root
        self.is_fs         = False
        self.drawing       = False
        self.last_xy       = None
        self.has_drawn     = False
        self.pen_r         = tk.IntVar(value=DEFAULT_PEN)
        self.var_save      = tk.BooleanVar(value=True)
        self.history       = []
        self.last_arr      = None
        self.last_pred     = None
        self.learn_total   = 0
        self.learn_correct = 0
        self.learn_fixed   = 0

        os.makedirs(SAVE_DIR, exist_ok=True)
        for c in NHAN:
            os.makedirs(os.path.join(SAVE_DIR, c), exist_ok=True)

        self._reset_pil()
        self.model = self._load_model()
        self._setup_win()
        self._build()
        self._bind()

    def _setup_win(self):
        self.root.title("NHAN DANG CHU CAI VIET TAY  A-Z  v6.0")
        self.root.configure(bg=BG)
        self.root.minsize(1080, 700)
        w, h = 1280, 830
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showerror("LOI", f"Khong tim thay '{MODEL_PATH}'.")
            self.root.destroy(); sys.exit(1)
        try:
            m = load_model(MODEL_PATH)
            print(f"[OK] {MODEL_PATH}")
            return m
        except Exception as e:
            messagebox.showerror("Loi model", str(e))
            self.root.destroy(); sys.exit(1)

    def _reset_pil(self):
        self.pil_img  = Image.new("RGB", (CANVAS_SZ, CANVAS_SZ), PIL_BG)
        self.pil_draw = ImageDraw.Draw(self.pil_img)

    # ══════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════
    def _build(self):
        self._build_title_bar()
        self._build_body()
        self._build_bottom_bar()

    # ── TITLE BAR ────────────────────────────────────────────────
    def _build_title_bar(self):
        bar = tk.Frame(self.root, bg=PANEL_DARK, height=80)
        bar.pack(fill="x"); bar.pack_propagate(False)

        # Top cyan line
        tk.Frame(bar, bg=CYAN, height=2).pack(fill="x")

        center = tk.Frame(bar, bg=PANEL_DARK)
        center.pack(expand=True, fill="both")

        # Decorative lines left
        left_dec = tk.Frame(center, bg=PANEL_DARK)
        left_dec.pack(side="left", fill="both", expand=True)
        tk.Frame(left_dec, bg=CYAN2, height=1).pack(
            fill="x", padx=20, pady=(18, 0))
        tk.Frame(left_dec, bg=CYAN_DARK, height=1).pack(
            fill="x", padx=20, pady=(4, 0))

        # Center title
        mid = tk.Frame(center, bg=PANEL_DARK)
        mid.pack(side="left")

        tk.Label(mid,
                 text="[  NHAN DANG CHU CAI VIET TAY A - Z  ]",
                 font=F_TITLE, bg=PANEL_DARK, fg=CYAN).pack(pady=(6, 0))
        tk.Label(mid,
                 text=">>  ONLINE LEARNING  |  DEEP LEARNING  |  v6.0  <<",
                 font=F_SUBTITLE, bg=PANEL_DARK, fg=CYAN2).pack()

        # Decorative lines right (mirror)
        right_dec = tk.Frame(center, bg=PANEL_DARK)
        right_dec.pack(side="left", fill="both", expand=True)
        tk.Frame(right_dec, bg=CYAN2, height=1).pack(
            fill="x", padx=20, pady=(18, 0))
        tk.Frame(right_dec, bg=CYAN_DARK, height=1).pack(
            fill="x", padx=20, pady=(4, 0))

        # Bottom cyan line
        tk.Frame(bar, bg=CYAN, height=2).pack(fill="x")

    # ── BODY ─────────────────────────────────────────────────────
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        self._build_left(body)

        # Separator
        sep = tk.Frame(body, bg=CYAN_DARK, width=2)
        sep.pack(side="left", fill="y", padx=8)

        self._build_right(body)

    # ── LEFT PANEL ───────────────────────────────────────────────
    def _build_left(self, parent):
        left = tk.Frame(parent, bg=BG)
        left.pack(side="left", fill="y")

        # Section label
        self._hud_label(left, "VUNG VE  (INPUT)", side="left")

        # HUD-bordered canvas container
        cv_outer = tk.Frame(left, bg=CYAN, padx=2, pady=2)
        cv_outer.pack(pady=(6, 0))
        cv_mid = tk.Frame(cv_outer, bg=CYAN2, padx=1, pady=1)
        cv_mid.pack()
        cv_inner = tk.Frame(cv_mid, bg=CV_BG)
        cv_inner.pack()

        self.canvas = tk.Canvas(cv_inner,
                                width=CANVAS_SZ, height=CANVAS_SZ,
                                bg=CV_BG, cursor="crosshair",
                                highlightthickness=0)
        self.canvas.pack()

        # Scanline overlay effect (subtle horizontal lines)
        for y in range(0, CANVAS_SZ, 6):
            self.canvas.create_line(0, y, CANVAS_SZ, y,
                                    fill="#0A1628", width=1)

        self.canvas.bind("<ButtonPress-1>",   self._press)
        self.canvas.bind("<B1-Motion>",       self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._release)

        # Brush control row
        brush_row = tk.Frame(left, bg=BG)
        brush_row.pack(fill="x", pady=(10, 0))

        # Pen icon + label
        icon_f = tk.Frame(brush_row, bg=CYAN_DARK, padx=6, pady=4)
        icon_f.pack(side="left")
        tk.Label(icon_f, text=" / ", font=F_HEAD,
                 bg=CYAN_DARK, fg=CYAN).pack(side="left")
        tk.Label(icon_f, text="CO BUT:", font=F_HEAD,
                 bg=CYAN_DARK, fg=CYAN).pack(side="left")

        slider_f = tk.Frame(brush_row, bg=BG)
        slider_f.pack(side="left", fill="x", expand=True, padx=(6, 0))

        self.slider = tk.Scale(slider_f, from_=1, to=30,
                               orient="horizontal", variable=self.pen_r,
                               showvalue=False, bg=PANEL_DARK, fg=CYAN,
                               troughcolor=CYAN_DARK,
                               activebackground=CYAN,
                               highlightthickness=0, bd=0,
                               command=self._pen_changed)
        self.slider.pack(fill="x")

        # Value + dot preview
        val_f = tk.Frame(brush_row, bg=CYAN_DARK, padx=6, pady=2)
        val_f.pack(side="left", padx=(6, 0))
        self.lbl_pen = tk.Label(val_f, text=f"{DEFAULT_PEN}px",
                                font=F_MONO, bg=CYAN_DARK, fg=CYAN, width=5)
        self.lbl_pen.pack(side="left")
        self.dot_cv = tk.Canvas(val_f, width=32, height=28,
                                bg=CYAN_DARK, highlightthickness=0)
        self.dot_cv.pack(side="left")
        self._draw_dot()

        # Preset buttons
        pre_f = tk.Frame(left, bg=BG)
        pre_f.pack(fill="x", pady=(6, 0))
        tk.Label(pre_f, text="[ PRESET:", font=F_TINY,
                 bg=BG, fg=MUTED).pack(side="left")
        for lbl, sz in [("XS", 3), ("S", 7), ("M", 14), ("L", 20), ("XL", 28)]:
            tk.Button(pre_f, text=lbl, font=("Courier New", 8, "bold"),
                      bg=CYAN_DARK, fg=CYAN,
                      activebackground=CYAN, activeforeground=BG,
                      relief="flat", cursor="hand2", width=3, pady=2,
                      command=lambda s=sz: self._set_pen(s)
                      ).pack(side="left", padx=(3, 0))
        tk.Label(pre_f, text=" ]", font=F_TINY, bg=BG, fg=MUTED).pack(side="left")

        # Action buttons row
        btn_f = tk.Frame(left, bg=BG)
        btn_f.pack(fill="x", pady=(14, 0))

        self.btn_pred = tk.Button(btn_f,
            text="  DU DOAN  ",
            font=("Courier New", 11, "bold"),
            bg=CYAN_DARK, fg=CYAN,
            activebackground=CYAN, activeforeground=BG,
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            command=self._predict)
        self.btn_pred.pack(side="left")

        tk.Frame(btn_f, bg=BG, width=8).pack(side="left")

        self.btn_clear_cv = tk.Button(btn_f,
            text="  XOA  ",
            font=("Courier New", 11, "bold"),
            bg=PANEL_DARK, fg=MUTED,
            activebackground=CARD_BG2, activeforeground=WHITE,
            relief="flat", cursor="hand2",
            padx=16, pady=8,
            command=self._clear)
        self.btn_clear_cv.pack(side="left")

        # Hint
        tk.Label(left, text="Ve, sau do nhan Du doan.  Del=Xoa  F11=Toan man",
                 font=F_TINY, bg=BG, fg=MUTED).pack(pady=(8, 0), anchor="w")

        # History
        self._hud_label(left, "LICH SU DU DOAN", side="left",
                        right_widget=lambda f: tk.Button(
                            f, text="[Xoa]", font=F_TINY,
                            bg=BG, fg=MUTED,
                            activebackground=PANEL_DARK, activeforeground=WHITE,
                            relief="flat", cursor="hand2",
                            command=self._clear_history).pack(side="right"))

        self.hist_outer = tk.Frame(left, bg=BG, height=88)
        self.hist_outer.pack(fill="x", pady=(6, 0))
        self.hist_outer.pack_propagate(False)
        self._render_history()

    # ── RIGHT PANEL (scrollable) ──────────────────────────────
    def _build_right(self, parent):
        right_outer = tk.Frame(parent, bg=BG)
        right_outer.pack(side="left", fill="both", expand=True)

        self.sf = ScrollFrame(right_outer, bg=BG)
        self.sf.pack(fill="both", expand=True)
        inn = self.sf.inner

        # ── 1. RESULT ────────────────────────────────────────
        self._hud_label(inn, "KET QUA NHAN DANG  (OUTPUT)")
        r_panel, r_inner = make_hud_frame(inn, bg=PANEL_BG,
                                          border_color=CYAN, padx=16, pady=14)
        r_panel.pack(fill="x", pady=(6, 0))

        char_row = tk.Frame(r_inner, bg=PANEL_BG)
        char_row.pack(fill="x")

        # Big glowing character box
        ch_box = tk.Frame(char_row, bg=CYAN_DARK,
                          padx=8, pady=4, relief="flat")
        ch_box.pack(side="left")
        self.lbl_char = tk.Label(ch_box, text="?",
                                 font=F_CHAR, bg=CYAN_DARK, fg=CYAN,
                                 width=2, anchor="center")
        self.lbl_char.pack()

        # Meta right side
        meta = tk.Frame(char_row, bg=PANEL_BG)
        meta.pack(side="left", fill="both", expand=True, padx=(14, 0))

        self.lbl_pct = tk.Label(meta, text="---",
                                font=F_PCT, bg=PANEL_BG, fg=CYAN)
        self.lbl_pct.pack(anchor="w")

        self.lbl_sub = tk.Label(meta,
                                text="Hay ve mot chu cai va nhan [Du doan]",
                                font=F_BODY, bg=PANEL_BG, fg=MUTED,
                                justify="left", wraplength=250)
        self.lbl_sub.pack(anchor="w", pady=(4, 0))

        # Confidence label row
        conf_hdr = tk.Frame(r_inner, bg=PANEL_BG)
        conf_hdr.pack(fill="x", pady=(12, 2))
        tk.Label(conf_hdr, text="DO TIN CAY:", font=F_HEAD,
                 bg=PANEL_BG, fg=MUTED).pack(side="left")
        self.lbl_conf_num = tk.Label(conf_hdr, text="",
                                     font=F_HEAD, bg=PANEL_BG, fg=CYAN)
        self.lbl_conf_num.pack(side="right")

        # Confidence bar
        conf_bg = tk.Frame(r_inner, bg=DIM, height=10)
        conf_bg.pack(fill="x"); conf_bg.pack_propagate(False)
        self.conf_bar = tk.Frame(conf_bg, bg=CYAN, height=10, width=0)
        self.conf_bar.place(x=0, y=0, relheight=1)

        # ── 2. TOP-5 ────────────────────────────────────────
        self._hud_label(inn, "TOP 5 XAC SUAT")
        t_panel, t_inner = make_hud_frame(inn, bg=PANEL_BG,
                                          border_color=CYAN2, padx=16, pady=12)
        t_panel.pack(fill="x", pady=(6, 0))
        self.chart_f = tk.Frame(t_inner, bg=PANEL_BG)
        self.chart_f.pack(fill="x")
        self._render_chart([])

        # ── 3. FEEDBACK ─────────────────────────────────────
        self._hud_label(inn, "AI DU DOAN DUNG KHONG?")
        fb_panel, fb_inner = make_hud_frame(inn, bg=PANEL_BG,
                                            border_color=PURPLE, padx=16, pady=14)
        fb_panel.pack(fill="x", pady=(6, 0))
        self._build_feedback(fb_inner)

    def _build_feedback(self, parent):
        # Subtitle
        desc_row = tk.Frame(parent, bg=PANEL_BG)
        desc_row.pack(fill="x")
        tk.Label(desc_row,
                 text="Danh gia de AI tu dong hoc them tu net chu cua ban  (Online Learning)",
                 font=F_TINY, bg=PANEL_BG, fg=MUTED,
                 justify="left", wraplength=420).pack(side="left", anchor="w")
        self.lbl_badge = tk.Label(desc_row, text="[ Da hoc: 0 ]",
                                  font=F_HEAD, bg=PURPLE_DARK, fg=PURPLE,
                                  padx=8, pady=3)
        self.lbl_badge.pack(side="right")

        # DUNG / SAI buttons
        fbrow = tk.Frame(parent, bg=PANEL_BG)
        fbrow.pack(fill="x", pady=(12, 0))

        self.btn_correct = tk.Button(fbrow,
            text="DUNG  (AI chinh xac)",
            font=F_BTN,
            bg=TEAL_DARK, fg=TEAL,
            activebackground="#00401A", activeforeground=TEAL,
            relief="flat", cursor="hand2",
            pady=12, state="disabled",
            command=self._on_correct)
        self.btn_correct.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self.btn_wrong = tk.Button(fbrow,
            text="SAI  (AI nham)",
            font=F_BTN,
            bg=RED_DARK, fg=RED_C,
            activebackground="#5A0018", activeforeground=RED_C,
            relief="flat", cursor="hand2",
            pady=12, state="disabled",
            command=self._on_wrong)
        self.btn_wrong.pack(side="left", fill="x", expand=True)

        # ── Correction panel (hidden by default) ────────────
        self.corr_panel = tk.Frame(parent, bg=CARD_BG2, padx=12, pady=12)
        # NOT packed yet – shown on _on_wrong()

        tk.Label(self.corr_panel,
                 text=">>  CHON CHU CAI DUNG  (de AI hoc lai):",
                 font=("Courier New", 9, "bold"),
                 bg=CARD_BG2, fg=AMBER).pack(anchor="w", pady=(0, 8))

        grid_f = tk.Frame(self.corr_panel, bg=CARD_BG2)
        grid_f.pack(fill="x")
        for i, c in enumerate(NHAN):
            r, col = divmod(i, 9)
            tk.Button(grid_f, text=c, width=3,
                      font=("Courier New", 10, "bold"),
                      bg=AMBER_DARK, fg=AMBER,
                      activebackground=AMBER, activeforeground=BG,
                      relief="flat", cursor="hand2",
                      command=lambda ch=c: self._learn_label(ch)
                      ).grid(row=r, column=col, padx=2, pady=2)

        tk.Checkbutton(self.corr_panel,
            text="Luu anh vao may  (du_lieu_thu_thap/)",
            variable=self.var_save,
            font=F_TINY, bg=CARD_BG2, fg=MUTED,
            activebackground=CARD_BG2, selectcolor=PANEL_DARK
        ).pack(anchor="w", pady=(10, 0))

        # Save also when DUNG
        tk.Checkbutton(parent,
            text="Luu anh vao may khi bam Dung",
            variable=self.var_save,
            font=F_TINY, bg=PANEL_BG, fg=MUTED,
            activebackground=PANEL_BG, selectcolor=PANEL_DARK
        ).pack(anchor="w", pady=(8, 0))

        # Status line
        self.lbl_status = tk.Label(parent, text="",
            font=("Courier New", 9, "bold"),
            bg=PANEL_BG, fg=TEAL, justify="center")
        self.lbl_status.pack(pady=(8, 0))

    # ── BOTTOM BAR ───────────────────────────────────────────────
    def _build_bottom_bar(self):
        tk.Frame(self.root, bg=CYAN, height=1).pack(fill="x")
        bar = tk.Frame(self.root, bg=PANEL_DARK, height=44)
        bar.pack(fill="x"); bar.pack_propagate(False)

        # Left: stat
        self.lbl_stat = tk.Label(bar,
            text="  Hoc: 0   Dung: 0   Da sua: 0  ",
            font=F_TINY, bg=PANEL_DARK, fg=CYAN2)
        self.lbl_stat.pack(side="left", padx=10)

        # Separator
        tk.Frame(bar, bg=CYAN_DARK, width=1).pack(side="left", fill="y", pady=10)

        # Center: save model
        self.btn_save_model = tk.Button(bar,
            text="  LUU MO HINH: CHUCAI.KERAS  ",
            font=F_TINY, bg=PANEL_DARK, fg=CYAN2,
            activebackground=CYAN_DARK, activeforeground=CYAN,
            relief="flat", cursor="hand2", padx=14,
            command=self._save_model)
        self.btn_save_model.pack(side="left", fill="y")

        # Separator
        tk.Frame(bar, bg=CYAN_DARK, width=1).pack(side="left", fill="y", pady=10)

        tk.Label(bar, text="  TRAI NGHIEM THUC TE (REAL-TIME)  ",
                 font=F_TINY, bg=PANEL_DARK, fg=MUTED).pack(side="left")

        # Right: fullscreen
        tk.Frame(bar, bg=CYAN_DARK, width=1).pack(side="right", fill="y", pady=10)
        self.btn_fs = tk.Button(bar,
            text="  TOAN MAN HINH (F11)  ",
            font=F_TINY, bg=PANEL_DARK, fg=CYAN2,
            activebackground=CYAN_DARK, activeforeground=CYAN,
            relief="flat", cursor="hand2",
            command=self._toggle_fs)
        self.btn_fs.pack(side="right")

        tk.Frame(self.root, bg=CYAN, height=2).pack(fill="x")

    # ── HUD section label ────────────────────────────────────────
    def _hud_label(self, parent, text, side="", right_widget=None):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(12, 0))
        # Left accent line
        tk.Frame(f, bg=CYAN, width=3).pack(side="left", fill="y")
        tk.Label(f, text=f"  {text}  ",
                 font=F_HEAD, bg=BG, fg=CYAN).pack(side="left")
        # Dashed line extending to right
        tk.Frame(f, bg=CYAN_DARK, height=1).pack(
            side="left", fill="x", expand=True, pady=6)
        if right_widget:
            right_widget(f)

    # ══════════════════════════════════════════════════════════════
    # MOUSE DRAWING
    # ══════════════════════════════════════════════════════════════
    def _press(self, e):
        self.drawing = True; self.last_xy = (e.x, e.y)
        r = self.pen_r.get()
        # Visual: bright white on dark canvas
        self.canvas.create_oval(e.x-r, e.y-r, e.x+r, e.y+r,
                                fill=CV_INK, outline=CV_INK)
        # PIL: black ink on white image (for model)
        self.pil_draw.ellipse([e.x-r, e.y-r, e.x+r, e.y+r], fill=PIL_INK)
        self.has_drawn = True
        if self.corr_panel.winfo_ismapped():
            self.corr_panel.pack_forget()

    def _drag(self, e):
        if not self.drawing or not self.last_xy: return
        x0, y0 = self.last_xy; r = self.pen_r.get()
        self.canvas.create_line(x0, y0, e.x, e.y, width=r*2,
                                fill=CV_INK, capstyle=tk.ROUND, smooth=True)
        self.pil_draw.line([x0, y0, e.x, e.y], fill=PIL_INK, width=r*2)
        self.last_xy = (e.x, e.y); self.has_drawn = True

    def _release(self, e):
        self.drawing = False; self.last_xy = None

    # ══════════════════════════════════════════════════════════════
    # BRUSH CONTROL
    # ══════════════════════════════════════════════════════════════
    def _pen_changed(self, *_):
        self.lbl_pen.config(text=f"{self.pen_r.get()}px")
        self._draw_dot()

    def _set_pen(self, s):
        self.pen_r.set(s); self._pen_changed()

    def _draw_dot(self):
        self.dot_cv.delete("all")
        r = min(self.pen_r.get(), 12)
        self.dot_cv.create_oval(16-r, 14-r, 16+r, 14+r,
                                fill=CYAN, outline="")

    # ══════════════════════════════════════════════════════════════
    # CLEAR
    # ══════════════════════════════════════════════════════════════
    def _clear(self):
        self.canvas.delete("all")
        # Redraw scanlines
        for y in range(0, CANVAS_SZ, 6):
            self.canvas.create_line(0, y, CANVAS_SZ, y,
                                    fill="#0A1628", width=1)
        self._reset_pil()
        self.has_drawn = False; self.last_arr = None; self.last_pred = None
        self.lbl_char.config(text="?", fg=CYAN)
        self.lbl_pct.config(text="---", fg=CYAN)
        self.lbl_conf_num.config(text="")
        self.lbl_sub.config(
            text="Hay ve mot chu cai va nhan [Du doan]", fg=MUTED)
        self.conf_bar.config(width=0)
        self._render_chart([])
        self._set_fb("disabled")
        if self.corr_panel.winfo_ismapped():
            self.corr_panel.pack_forget()
        self.lbl_status.config(text="")

    def _clear_history(self):
        self.history.clear(); self._render_history()

    def _toggle_fs(self):
        self.is_fs = not self.is_fs
        self.root.attributes("-fullscreen", self.is_fs)
        self.btn_fs.config(
            text=("  THOAT TOAN MAN (ESC)  " if self.is_fs
                  else "  TOAN MAN HINH (F11)  "))

    # ══════════════════════════════════════════════════════════════
    # PREPROCESS + PREDICT
    # ══════════════════════════════════════════════════════════════
    def _pre(self):
        g = self.pil_img.convert("L").resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        a = np.array(g, dtype=np.float32)
        a = (255.0 - a) / 255.0       # invert + normalize
        return a.reshape(1, IMG_SIZE * IMG_SIZE)

    def _predict(self):
        if not self.has_drawn:
            messagebox.showwarning("Chua ve gi",
                "Vui long ve mot chu cai truoc khi nhan dang."); return
        try:
            t0    = time.perf_counter()
            arr   = self._pre()
            probs = self.model.predict(arr, verbose=0)[0]
            ms    = (time.perf_counter() - t0) * 1000

            idx  = int(np.argmax(probs)); char = NHAN[idx]
            conf = float(probs[idx]) * 100
            clr  = TEAL if conf >= 70 else (AMBER if conf >= 40 else RED_C)

            self.last_arr = arr.copy(); self.last_pred = idx

            self.lbl_char.config(text=char, fg=CYAN)
            self.lbl_pct.config(text=f"{conf:.1f}%", fg=clr)
            self.lbl_conf_num.config(text=f"{conf:.1f}%  ({ms:.0f} ms)", fg=clr)
            self.lbl_sub.config(
                text=f"Ky tu nhan dang: [ {char} ]   Thoi gian: {ms:.0f} ms",
                fg=MUTED)

            self.root.update_idletasks()
            tw = self.conf_bar.master.winfo_width()
            self.conf_bar.config(width=max(1, int(tw * conf/100)), bg=clr)

            top5 = [(NHAN[i], float(probs[i])*100)
                    for i in np.argsort(probs)[::-1][:5]]
            self._render_chart(top5)

            self.history.insert(0, (char, conf))
            self.history = self.history[:MAX_HIST]
            self._render_history()

            self._set_fb("normal")
            if self.corr_panel.winfo_ismapped():
                self.corr_panel.pack_forget()
            self.lbl_status.config(text="")
            self.sf.scroll_top()

            print(f"[OK] '{char}' {conf:.1f}% {ms:.0f}ms")
        except Exception as ex:
            messagebox.showerror("Loi du doan", str(ex))

    # ══════════════════════════════════════════════════════════════
    # FEEDBACK & ONLINE LEARNING
    # ══════════════════════════════════════════════════════════════
    def _on_correct(self):
        if self.last_arr is None: return
        self.learn_correct += 1
        self._do_learn(self.last_arr, self.last_pred,
                       NHAN[self.last_pred], is_fix=False)

    def _on_wrong(self):
        if self.last_arr is None: return
        self._set_fb("disabled")
        self.corr_panel.pack(fill="x", pady=(12, 0))
        self.root.after(60, lambda: self.sf.scroll_bot())

    def _learn_label(self, correct_char):
        if self.last_arr is None: return
        self.learn_fixed += 1
        self.corr_panel.pack_forget()
        self._do_learn(self.last_arr, NHAN.index(correct_char),
                       correct_char, is_fix=True)

    def _do_learn(self, arr, idx, char, is_fix):
        try:
            lbl = to_categorical([idx], num_classes=26)
            self.model.fit(arr, lbl, epochs=1, verbose=0)
            self.learn_total += 1

            saved = self._save_img(char) if self.var_save.get() else ""

            if is_fix:
                pred = NHAN[self.last_pred]
                msg  = f"[ HOC XONG ]  AI nham '{pred}'  =>  da hoc nhan '{char}'"
                clr  = AMBER
            else:
                msg  = f"[ HOC XONG ]  Cu cung nhan '{char}'  –  AI se tot hon!"
                clr  = TEAL

            self.lbl_status.config(text=msg, fg=clr)
            self.lbl_badge.config(text=f"[ Da hoc: {self.learn_total} ]")
            self.lbl_stat.config(
                text=f"  Hoc: {self.learn_total}   Dung: {self.learn_correct}   Da sua: {self.learn_fixed}  ")
            self._set_fb("disabled")

            detail = f"\n\nDa luu anh: {saved}" if saved else ""
            messagebox.showinfo("Hoc thanh cong!", f"{msg}{detail}")
        except Exception as ex:
            messagebox.showerror("Loi hoc", str(ex))

    def _save_img(self, ch):
        folder = os.path.join(SAVE_DIR, ch)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.pil_img.save(os.path.join(folder, f"{ts}_orig.png"))
        a2 = (self.last_arr.reshape(IMG_SIZE, IMG_SIZE)*255).astype(np.uint8)
        Image.fromarray(a2, "L").save(os.path.join(folder, f"{ts}_28x28.png"))
        return folder

    def _save_model(self):
        if self.learn_total == 0:
            messagebox.showinfo("Chua hoc lan nao",
                "Chua co lan hoc nao. Hay danh gia du doan truoc."); return
        try:
            self.model.save(MODEL_PATH)
            messagebox.showinfo("Luu thanh cong!",
                f"Da luu: '{MODEL_PATH}'\n"
                f"Tong hoc: {self.learn_total}  |  Dung: {self.learn_correct}  |  Da sua: {self.learn_fixed}")
        except Exception as ex:
            messagebox.showerror("Loi luu", str(ex))

    def _set_fb(self, st):
        self.btn_correct.config(state=st)
        self.btn_wrong.config(state=st)

    # ══════════════════════════════════════════════════════════════
    # TOP-5 CHART
    # ══════════════════════════════════════════════════════════════
    def _render_chart(self, data):
        for w in self.chart_f.winfo_children(): w.destroy()
        if not data:
            tk.Label(self.chart_f, text="--  chua co du lieu  --",
                     font=F_TINY, bg=PANEL_BG, fg=DIM).pack(anchor="w"); return
        for rank, (ltr, pct) in enumerate(data):
            row = tk.Frame(self.chart_f, bg=PANEL_BG)
            row.pack(fill="x", pady=3)
            fc = CYAN if rank == 0 else WHITE
            tk.Label(row, text=ltr, width=2,
                     font=("Courier New", 10, "bold"),
                     bg=PANEL_BG, fg=fc).pack(side="left")
            bar_bg = tk.Frame(row, bg=DIM, height=16)
            bar_bg.pack(side="left", fill="x", expand=True, padx=(8, 8))
            bar_bg.pack_propagate(False)
            bc = CYAN if rank == 0 else CYAN2
            bar_bg.after(20, lambda w=bar_bg, p=pct, c=bc:
                         self._fill_bar(w, p, c))
            tk.Label(row, text=f"{pct:.1f}%", width=7,
                     font=F_MONO, bg=PANEL_BG, fg=MUTED).pack(side="left")

    def _fill_bar(self, w, pct, color):
        try:
            w.update_idletasks()
            fw = max(2, int(w.winfo_width() * pct / 100))
            tk.Frame(w, bg=color, height=16, width=fw).place(
                x=0, y=0, relheight=1)
        except Exception: pass

    # ══════════════════════════════════════════════════════════════
    # HISTORY
    # ══════════════════════════════════════════════════════════════
    def _render_history(self):
        for w in self.hist_outer.winfo_children(): w.destroy()
        if not self.history:
            tk.Label(self.hist_outer, text="Chua co ket qua nao",
                     font=F_TINY, bg=BG, fg=DIM).pack(anchor="w"); return
        row = tk.Frame(self.hist_outer, bg=BG)
        row.pack(anchor="w")
        for char, conf in self.history:
            cell = tk.Frame(row, bg=CYAN_DARK, padx=7, pady=4)
            cell.pack(side="left", padx=(0, 4))
            tk.Label(cell, text=char,
                     font=("Courier New", 18, "bold"),
                     bg=CYAN_DARK, fg=CYAN).pack()
            c = TEAL if conf >= 70 else (AMBER if conf >= 40 else RED_C)
            tk.Label(cell, text=f"{conf:.0f}%",
                     font=("Courier New", 7),
                     bg=CYAN_DARK, fg=c).pack()

    # ══════════════════════════════════════════════════════════════
    # KEY BINDINGS
    # ══════════════════════════════════════════════════════════════
    def _bind(self):
        self.root.bind("<Return>", lambda e: self._predict())
        self.root.bind("<Delete>", lambda e: self._clear())
        self.root.bind("<Escape>", lambda e: (
            self._toggle_fs() if self.is_fs else self._clear()))
        self.root.bind("<F11>",    lambda e: self._toggle_fs())
        self.root.bind("<Prior>",  lambda e: self._set_pen(
            min(self.pen_r.get()+2, 30)))
        self.root.bind("<Next>",   lambda e: self._set_pen(
            max(self.pen_r.get()-2, 1)))


# ══════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════
def main():
    root = tk.Tk()
    try: root.iconbitmap("icon.ico")
    except Exception: pass
    App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
