"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        MNIST El Yazısı Hesap Makinesi — Ana GUI Uygulaması v2              ║
║                                                                              ║
║  📝 İşleyiş:                                                               ║
║     1. Sol panel: 1-3 rakam çizilir (mor kutu) → soldan sağa okunur      ║
║     2. Orta panel: Matematiksel operatör (+, −, ×, ÷) çizilir (amber)     ║
║     3. Sağ panel: 1-3 rakam çizilir (mor kutu)                           ║
║     4. Her alanda fare kaldırılınca 1 saniye sonra otomatik tanıma        ║
║     5. Tüm tahminler hazır → sonuç otomatik hesaplanır                   ║
║     6. Güven skoru: yeşil (≥%90) | sarı (%70-89) | kırmızı (<%70)        ║
║                                                                              ║
║  🎯 Amaç: Derin öğrenme + GUI tasarımını pratik uygulamada göster        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from PIL import Image, ImageDraw
import os
import subprocess
import sys

try:
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

DIGIT_MODEL_PATH = "mnist_model.keras"
OP_MODEL_PATH    = "operator_model.keras"
OPERATORS        = ['+', '−', '×', '/']   # indeks → sembol

# ─── Renk Paleti ──────────────────────────────────────────────────────────────
BG_DARK    = "#1E1E2E"
BG_PANEL   = "#2A2A3E"
BG_CANVAS  = "#0D0D1A"
DIG_ACCENT = "#7C3AED"   # rakam yuvaları — mor
OP_ACCENT  = "#D97706"   # operatör yuvası — amber
RESULT_CLR = "#10B981"   # sonuç — yeşil
BTN_RED    = "#EF4444"
TEXT_LIGHT = "#E2E8F0"
TEXT_DIM   = "#94A3B8"

SLOT_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]
CONF_LOW    = "#EF4444"   # < %70
CONF_MID    = "#F59E0B"   # %70–89
CONF_HIGH   = "#10B981"   # ≥ %90



# ─────────────────────────────────────────────────────────────────────────────
# Temel Çizim Alanı
# ─────────────────────────────────────────────────────────────────────────────
# Temel Çizim Alanı (Canvas)
# ─────────────────────────────────────────────────────────────────────────────
# 📌 ÖNEMLİ: DrawCanvas — Kullanıcının çizim yaptığı widget
class DrawCanvas(tk.Canvas):
    """
    Çizim Yüzeyi (160×160 piksel)
    
    ÖZELLIKLER:
    • accent_color  → Kenarlık rengi (mor rakam / amber operatör)
    • brush_radius  → Fırça kalınlığı (piksel)
    • on_idle       → Fare durağanlaştıktan 1 sn sonra çağrılan callback
    
    İÇ VERİ:
    • _pil_img: PIL Image nesnesi (çizim yapılıyor)
    • _pil_drw: PIL ImageDraw (çizim fonksiyonları)
    • _timer: threading.Timer (1 sn sonra otomatik tahmin)
    
    ÇALIŞMA ŞEMASI:
    1. Mouse down (_start)   → Son konum kaydet
    2. Mouse move (_paint)   → Çizgi çiz (Tkinter + PIL)
    3. Mouse up (_release)   → Timer başlat
    4. 1 sn bekleme...
    5. Timer çıkışında _fire() → on_idle() callback çağır → tahmin et
    """
    SIZE = 160  # 160×160 piksel çizim alanı (28×28'e yeniden boyutlandırılacak)

    def __init__(self, master, accent_color, brush_radius=9, on_idle=None, **kw):
        super().__init__(
            master,
            width=self.SIZE, height=self.SIZE,
            bg=BG_CANVAS, cursor="crosshair",
            highlightthickness=3, highlightbackground=accent_color,
            **kw,
        )
        self._brush   = brush_radius
        self._on_idle = on_idle
        self._pil_img = Image.new("L", (self.SIZE, self.SIZE), 0)
        self._pil_drw = ImageDraw.Draw(self._pil_img)
        self._last    = None
        self._timer   = None

        self.bind("<Button-1>",        self._start)
        self.bind("<B1-Motion>",       self._paint)
        self.bind("<ButtonRelease-1>", self._release)

    def _start(self, e):
        self._last = (e.x, e.y)

    def _paint(self, e):
        r = self._brush
        self.create_oval(e.x - r, e.y - r, e.x + r, e.y + r,
                         fill="white", outline="white")
        self._pil_drw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
        if self._last:
            self.create_line(*self._last, e.x, e.y,
                             fill="white", width=r * 2,
                             capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self._pil_drw.line([*self._last, e.x, e.y], fill=255, width=r * 2)
        self._last = (e.x, e.y)
        self._arm_timer()

    def _release(self, e):
        self._last = None
        self._arm_timer()

    def _arm_timer(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(1.0, self._fire)
        self._timer.start()

    def _fire(self):
        if self._on_idle:
            self.after(0, self._on_idle)

    def get_array(self) -> np.ndarray:
        """28×28 float32 [0,1] dizisi."""
        img = self._pil_img.resize((28, 28), Image.LANCZOS)
        return np.array(img, dtype="float32") / 255.0

    def is_empty(self) -> bool:
        return np.max(np.array(self._pil_img)) == 0

    def clear(self):
        if self._timer:
            self._timer.cancel()
        self.delete("all")
        self._pil_img = Image.new("L", (self.SIZE, self.SIZE), 0)
        self._pil_drw = ImageDraw.Draw(self._pil_img)
        self._last = None


# ─────────────────────────────────────────────────────────────────────────────
# Rakam Yuvası  (DrawCanvas + tahmin etiketi + güven skoru)
# ─────────────────────────────────────────────────────────────────────────────
class DigitSlot(tk.Frame):
    """Tek rakam çizim yuvası."""

    def __init__(self, master, slot_index: int, on_idle=None, **kw):
        super().__init__(master, bg=BG_DARK, padx=4, pady=4, **kw)
        self.slot_index = slot_index
        self.digit      = None
        self.confidence = 0.0

        color = SLOT_COLORS[slot_index % len(SLOT_COLORS)]

        tk.Label(self, text=f"Rakam {slot_index + 1}",
                 bg=BG_DARK, fg=color,
                 font=("Segoe UI", 8, "bold")).pack()

        self.canvas = DrawCanvas(self, DIG_ACCENT, brush_radius=9, on_idle=on_idle)
        self.canvas.pack()

        self._pred_var = tk.StringVar(value="?")
        tk.Label(self, textvariable=self._pred_var,
                 bg=BG_DARK, fg=color,
                 font=("Segoe UI", 22, "bold")).pack()

        self._conf_var = tk.StringVar(value="")
        self._conf_lbl = tk.Label(self, textvariable=self._conf_var,
                                  bg=BG_DARK, fg=TEXT_DIM,
                                  font=("Segoe UI", 8))
        self._conf_lbl.pack()

    def set_result(self, digit: int, confidence: float):
        self.digit      = digit
        self.confidence = confidence
        self._pred_var.set(str(digit))
        pct = confidence * 100
        self._conf_var.set(f"%{pct:.1f}")
        self._conf_lbl.config(fg=_conf_color(pct))

    def clear(self):
        self.canvas.clear()
        self.digit      = None
        self.confidence = 0.0
        self._pred_var.set("?")
        self._conf_var.set("")
        self._conf_lbl.config(fg=TEXT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Operatör Yuvası  (amber renkli — rakam kutularından görsel olarak ayrışır)
# ─────────────────────────────────────────────────────────────────────────────
class OperatorSlot(tk.Frame):
    """Operatör çizim yuvası."""

    def __init__(self, master, on_idle=None, **kw):
        super().__init__(master, bg=BG_DARK, padx=6, pady=4, **kw)
        self.operator   = None
        self.confidence = 0.0

        tk.Label(self, text="Operatör",
                 bg=BG_DARK, fg=OP_ACCENT,
                 font=("Segoe UI", 9, "bold")).pack()
        tk.Label(self, text="+   −   ×   /",
                 bg=BG_DARK, fg=TEXT_DIM,
                 font=("Segoe UI", 8)).pack()

        # Amber kenarlık + biraz daha ince fırça (8px) → operatör çizgileri belirgin
        self.canvas = DrawCanvas(self, OP_ACCENT, brush_radius=8, on_idle=on_idle)
        self.canvas.pack()

        self._pred_var = tk.StringVar(value="?")
        tk.Label(self, textvariable=self._pred_var,
                 bg=BG_DARK, fg=OP_ACCENT,
                 font=("Segoe UI", 22, "bold")).pack()

        self._conf_var = tk.StringVar(value="")
        self._conf_lbl = tk.Label(self, textvariable=self._conf_var,
                                  bg=BG_DARK, fg=TEXT_DIM,
                                  font=("Segoe UI", 8))
        self._conf_lbl.pack()

    def set_result(self, op_str: str, confidence: float):
        self.operator   = op_str
        self.confidence = confidence
        self._pred_var.set(op_str)
        pct = confidence * 100
        self._conf_var.set(f"%{pct:.1f}")
        self._conf_lbl.config(fg=_conf_color(pct))

    def clear(self):
        self.canvas.clear()
        self.operator   = None
        self.confidence = 0.0
        self._pred_var.set("?")
        self._conf_var.set("")
        self._conf_lbl.config(fg=TEXT_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Yardımcı
# ─────────────────────────────────────────────────────────────────────────────
def _conf_color(pct: float) -> str:
    if pct >= 90:
        return CONF_HIGH
    if pct >= 70:
        return CONF_MID
    return CONF_LOW


# ─────────────────────────────────────────────────────────────────────────────
# Ana Pencere
# ─────────────────────────────────────────────────────────────────────────────
class App(tk.Tk):
    MAX_SIDE = 3   # her tarafta en fazla 3 basamak

    def __init__(self):
        super().__init__()
        self.title("MNIST El Yazısı Hesap Makinesi")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)

        self.digit_model = None
        self.op_model    = None
        self._load_models()

        self.left_count  = tk.IntVar(value=1)
        self.right_count = tk.IntVar(value=1)
        self.status_var  = tk.StringVar(value="Hazır.")
        self.result_var  = tk.StringVar(value="")
        self.expr_var    = tk.StringVar(value="")

        self._left_slots:  list = []
        self._right_slots: list = []
        self._op_slot     = None

        self._build_ui()
        self._rebuild_left()
        self._rebuild_right()

    # ─── Model Yükleme ────────────────────────────────────────────────────────
    def _load_models(self):
        if not TF_AVAILABLE:
            messagebox.showerror("TensorFlow", "TensorFlow yüklü değil.")
            return
        if os.path.exists(DIGIT_MODEL_PATH):
            self.digit_model = keras.models.load_model(DIGIT_MODEL_PATH)
        else:
            messagebox.showwarning("Model Yok",
                f"{DIGIT_MODEL_PATH} bulunamadı.\nÖnce train_model.py çalıştırın.")
        if os.path.exists(OP_MODEL_PATH):
            self.op_model = keras.models.load_model(OP_MODEL_PATH)
        else:
            messagebox.showwarning("Operatör Modeli Yok",
                f"{OP_MODEL_PATH} bulunamadı.\nÖnce train_operator.py çalıştırın.")

    # ─── UI İnşası ────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Başlık
        hdr = tk.Frame(self, bg=DIG_ACCENT, pady=8)
        hdr.pack(fill="x")
        tk.Label(hdr, text="MNIST El Yazısı Hesap Makinesi",
                 font=("Segoe UI", 16, "bold"),
                 bg=DIG_ACCENT, fg="white").pack()
        tk.Label(hdr,
                 text="Sol sayıyı · operatörü · sağ sayıyı çizin — 1 sn sonra otomatik tanınır",
                 font=("Segoe UI", 8), bg=DIG_ACCENT, fg="#DDD6FE").pack()

        # Üç bölüm
        sections = tk.Frame(self, bg=BG_DARK)
        sections.pack(fill="both", expand=True, padx=12, pady=10)

        # — Sol sayı —
        self._left_wrap = tk.LabelFrame(
            sections, text=" Sol Sayı ",
            bg=BG_DARK, fg=DIG_ACCENT,
            font=("Segoe UI", 10, "bold"), padx=6, pady=6,
        )
        self._left_wrap.pack(side="left", fill="y")

        ctrl_l = tk.Frame(self._left_wrap, bg=BG_DARK)
        ctrl_l.pack(fill="x", pady=(0, 5))
        tk.Label(ctrl_l, text="Basamak:", bg=BG_DARK, fg=TEXT_DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        for n in range(1, self.MAX_SIDE + 1):
            tk.Radiobutton(ctrl_l, text=str(n), variable=self.left_count, value=n,
                           bg=BG_DARK, fg=TEXT_LIGHT, selectcolor=DIG_ACCENT,
                           activebackground=BG_DARK,
                           command=self._rebuild_left).pack(side="left")

        self._left_slots_frame = tk.Frame(self._left_wrap, bg=BG_DARK)
        self._left_slots_frame.pack()

        self._left_num_lbl = tk.Label(self._left_wrap, text="—",
                                      bg=BG_DARK, fg=DIG_ACCENT,
                                      font=("Segoe UI", 20, "bold"))
        self._left_num_lbl.pack(pady=(6, 0))

        # — Operatör —
        op_wrap = tk.LabelFrame(
            sections, text=" Operatör ",
            bg=BG_DARK, fg=OP_ACCENT,
            font=("Segoe UI", 10, "bold"), padx=8, pady=6,
        )
        op_wrap.pack(side="left", fill="y", padx=10)

        self._op_slot = OperatorSlot(op_wrap, on_idle=self._auto_op)
        self._op_slot.pack()

        # — Sağ sayı —
        self._right_wrap = tk.LabelFrame(
            sections, text=" Sağ Sayı ",
            bg=BG_DARK, fg=DIG_ACCENT,
            font=("Segoe UI", 10, "bold"), padx=6, pady=6,
        )
        self._right_wrap.pack(side="left", fill="y")

        ctrl_r = tk.Frame(self._right_wrap, bg=BG_DARK)
        ctrl_r.pack(fill="x", pady=(0, 5))
        tk.Label(ctrl_r, text="Basamak:", bg=BG_DARK, fg=TEXT_DIM,
                 font=("Segoe UI", 8)).pack(side="left")
        for n in range(1, self.MAX_SIDE + 1):
            tk.Radiobutton(ctrl_r, text=str(n), variable=self.right_count, value=n,
                           bg=BG_DARK, fg=TEXT_LIGHT, selectcolor=DIG_ACCENT,
                           activebackground=BG_DARK,
                           command=self._rebuild_right).pack(side="left")

        self._right_slots_frame = tk.Frame(self._right_wrap, bg=BG_DARK)
        self._right_slots_frame.pack()

        self._right_num_lbl = tk.Label(self._right_wrap, text="—",
                                       bg=BG_DARK, fg=DIG_ACCENT,
                                       font=("Segoe UI", 20, "bold"))
        self._right_num_lbl.pack(pady=(6, 0))

        # Buton satırı
        btn_row = tk.Frame(self, bg=BG_DARK, pady=6)
        btn_row.pack(fill="x", padx=12)

        self._btn(btn_row, "Tümünü Tahmin Et", self._predict_all,
                  DIG_ACCENT, "white").pack(side="left", padx=4)
        self._btn(btn_row, "Hesapla", self._calculate,
                  RESULT_CLR, "white").pack(side="left", padx=4)
        self._btn(btn_row, "Temizle", self._clear_all,
                  BTN_RED, "white").pack(side="left", padx=4)
        self._btn(btn_row, "📊 Confusion Matrix", self._show_cm,
                  "#8B5CF6", "white").pack(side="right", padx=4)
        self._btn(btn_row, "📈 Karşılaştır", self._show_cmp,
                  "#06B6D4", "white").pack(side="right", padx=4)

        # Sonuç şeridi
        res_bar = tk.Frame(self, bg=BG_PANEL, pady=10, padx=15)
        res_bar.pack(fill="x", padx=12)
        tk.Label(res_bar, textvariable=self.expr_var,
                 bg=BG_PANEL, fg=TEXT_DIM,
                 font=("Segoe UI", 14)).pack(side="left")
        tk.Label(res_bar, textvariable=self.result_var,
                 bg=BG_PANEL, fg=RESULT_CLR,
                 font=("Segoe UI", 24, "bold")).pack(side="left", padx=12)

        # Durum çubuğu
        status_bar = tk.Frame(self, bg=BG_PANEL, pady=3)
        status_bar.pack(fill="x", side="bottom")
        tk.Label(status_bar, textvariable=self.status_var,
                 bg=BG_PANEL, fg=TEXT_DIM, font=("Segoe UI", 8)).pack()

    @staticmethod
    def _btn(parent, text, cmd, bg, fg):
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg, font=("Segoe UI", 9, "bold"),
                         relief="flat", padx=10, pady=6, cursor="hand2",
                         activebackground=bg, activeforeground=fg)

    # ─── Slot Yönetimi ────────────────────────────────────────────────────────
    def _rebuild_left(self):
        for w in self._left_slots_frame.winfo_children():
            w.destroy()
        self._left_slots.clear()
        for i in range(self.left_count.get()):
            s = DigitSlot(self._left_slots_frame, i,
                          on_idle=lambda idx=i: self._auto_digit("left", idx))
            s.pack(side="left")
            self._left_slots.append(s)
        self._left_num_lbl.config(text="—")
        self._reset_result()

    def _rebuild_right(self):
        for w in self._right_slots_frame.winfo_children():
            w.destroy()
        self._right_slots.clear()
        for i in range(self.right_count.get()):
            s = DigitSlot(self._right_slots_frame, i,
                          on_idle=lambda idx=i: self._auto_digit("right", idx))
            s.pack(side="left")
            self._right_slots.append(s)
        self._right_num_lbl.config(text="—")
        self._reset_result()

    # ─── Tahmin ───────────────────────────────────────────────────────────────
    def _infer_digit(self, slot: DigitSlot):
        if not self.digit_model or slot.canvas.is_empty():
            return
        arr   = slot.canvas.get_array().reshape(1, 28, 28, 1)
        probs = self.digit_model.predict(arr, verbose=0)[0]
        slot.set_result(int(np.argmax(probs)), float(probs.max()))

    def _infer_operator(self):
        if not self.op_model or self._op_slot.canvas.is_empty():
            return
        arr   = self._op_slot.canvas.get_array().reshape(1, 28, 28, 1)
        probs = self.op_model.predict(arr, verbose=0)[0]
        idx   = int(np.argmax(probs))
        self._op_slot.set_result(OPERATORS[idx], float(probs[idx]))

    def _auto_digit(self, side: str, idx: int):
        slots = self._left_slots if side == "left" else self._right_slots
        if idx < len(slots):
            self._infer_digit(slots[idx])
        self._update_num_label(side)
        self._try_auto_calc()
        tag = slots[idx].digit if idx < len(slots) else "?"
        self.status_var.set(
            f"Otomatik tahmin: {'Sol' if side == 'left' else 'Sağ'} Rakam {idx+1} → {tag}"
        )

    def _auto_op(self):
        self._infer_operator()
        self._try_auto_calc()
        self.status_var.set(f"Operatör tahmin edildi → {self._op_slot.operator}")

    def _predict_all(self):
        for s in self._left_slots:
            self._infer_digit(s)
        self._infer_operator()
        for s in self._right_slots:
            self._infer_digit(s)
        self._update_num_label("left")
        self._update_num_label("right")
        self._try_auto_calc()
        self.status_var.set("Tüm tahminler tamamlandı.")

    # ─── Sayı Birleştirme ─────────────────────────────────────────────────────
    def _update_num_label(self, side: str):
        slots = self._left_slots if side == "left" else self._right_slots
        lbl   = self._left_num_lbl if side == "left" else self._right_num_lbl
        digits = [s.digit for s in slots if s.digit is not None]
        lbl.config(text="".join(str(d) for d in digits) if digits else "—")

    def _assembled(self, side: str):
        slots  = self._left_slots if side == "left" else self._right_slots
        digits = [s.digit for s in slots if s.digit is not None]
        if not digits or len(digits) != len(slots):
            return None
        return int("".join(str(d) for d in digits))

    # ─── Hesaplama ────────────────────────────────────────────────────────────
    def _try_auto_calc(self):
        n1 = self._assembled("left")
        op = self._op_slot.operator
        n2 = self._assembled("right")
        if n1 is not None and op is not None and n2 is not None:
            self._do_calc(n1, op, n2)

    def _calculate(self):
        n1 = self._assembled("left")
        op = self._op_slot.operator
        n2 = self._assembled("right")
        if n1 is None:
            messagebox.showinfo("Eksik", "Sol sayıyı çizin ve tahmin ettirin.")
            return
        if op is None:
            messagebox.showinfo("Eksik", "Operatörü çizin ve tahmin ettirin.")
            return
        if n2 is None:
            messagebox.showinfo("Eksik", "Sağ sayıyı çizin ve tahmin ettirin.")
            return
        self._do_calc(n1, op, n2)

    def _do_calc(self, n1: int, op: str, n2: int):
        try:
            if   op == '+': res = n1 + n2
            elif op == '−': res = n1 - n2
            elif op == '×': res = n1 * n2
            elif op == '/':
                if n2 == 0:
                    self.expr_var.set(f"{n1} / {n2}  =")
                    self.result_var.set("Sıfıra bölme!")
                    return
                res = n1 / n2
            else:
                return
            display = int(res) if isinstance(res, float) and res.is_integer() else round(res, 6)
            self.expr_var.set(f"{n1}  {op}  {n2}  =")
            self.result_var.set(str(display))
            self.status_var.set(f"Sonuç: {n1} {op} {n2} = {display}")
        except Exception as exc:
            messagebox.showerror("Hata", str(exc))

    # ─── Temizle ──────────────────────────────────────────────────────────────
    def _clear_all(self):
        for s in self._left_slots:
            s.clear()
        self._op_slot.clear()
        for s in self._right_slots:
            s.clear()
        self._left_num_lbl.config(text="—")
        self._right_num_lbl.config(text="—")
        self._reset_result()
        self.status_var.set("Temizlendi.")

    def _reset_result(self):
        self.result_var.set("")
        self.expr_var.set("")

    # ─── Harici Görselleştirmeler ─────────────────────────────────────────────
    def _show_cm(self):
        subprocess.Popen([sys.executable, "show_confusion_matrix.py"])

    def _show_cmp(self):
        subprocess.Popen([sys.executable, "show_comparison.py"])


# ─── Giriş Noktası ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().mainloop()

