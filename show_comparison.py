"""
Augmentation var/yok model karşılaştırma grafiği (app.py'den çağrılır)
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow import keras
from tensorflow.keras.datasets import mnist

MODEL_AUG    = "mnist_model.keras"
MODEL_NO_AUG = "mnist_model_no_aug.keras"
HIST_AUG     = "history.json"
HIST_NO_AUG  = "history_no_aug.json"


def load_history(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    h_aug    = load_history(HIST_AUG)
    h_no_aug = load_history(HIST_NO_AUG)

    if h_aug is None or h_no_aug is None:
        print("HATA: Geçmiş dosyaları bulunamadı. Önce train_model.py çalıştırın.")
        return

    # Test doğruluklarını hesapla
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]
    y_cat  = keras.utils.to_categorical(y_test, 10)

    acc_aug = acc_no = None
    if os.path.exists(MODEL_AUG):
        m = keras.models.load_model(MODEL_AUG)
        _, acc_aug = m.evaluate(x_test, y_cat, verbose=0)
    if os.path.exists(MODEL_NO_AUG):
        m2 = keras.models.load_model(MODEL_NO_AUG)
        _, acc_no = m2.evaluate(x_test, y_cat, verbose=0)

    # ── Grafik ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Augmentation Var vs Yok — Performans Karşılaştırması",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    C_AUG   = "#2196F3"
    C_NOAUG = "#FF5722"

    def _plot(ax, aug_vals, no_vals, title, ylabel, higher_is_better=True):
        epochs_a = range(1, len(aug_vals) + 1)
        epochs_n = range(1, len(no_vals) + 1)
        ax.plot(epochs_a, aug_vals,  color=C_AUG,   linewidth=2, marker="o", markersize=3, label="Augmentation VAR")
        ax.plot(epochs_n, no_vals,   color=C_NOAUG,  linewidth=2, marker="s", markersize=3, label="Augmentation YOK")
        best_a = max(aug_vals) if higher_is_better else min(aug_vals)
        best_n = max(no_vals)  if higher_is_better else min(no_vals)
        ax.axhline(best_a, color=C_AUG,   linestyle="--", alpha=0.35, linewidth=1)
        ax.axhline(best_n, color=C_NOAUG, linestyle="--", alpha=0.35, linewidth=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Eğitim doğruluğu
    _plot(fig.add_subplot(gs[0, 0]),
          h_aug["accuracy"], h_no_aug["accuracy"],
          "Eğitim Doğruluğu", "Accuracy")
    # Eğitim kaybı
    _plot(fig.add_subplot(gs[0, 1]),
          h_aug["loss"], h_no_aug["loss"],
          "Eğitim Kaybı", "Loss", higher_is_better=False)
    # Doğrulama doğruluğu
    _plot(fig.add_subplot(gs[1, 0]),
          h_aug["val_accuracy"], h_no_aug["val_accuracy"],
          "Doğrulama Doğruluğu", "Accuracy")
    # Doğrulama kaybı
    _plot(fig.add_subplot(gs[1, 1]),
          h_aug["val_loss"], h_no_aug["val_loss"],
          "Doğrulama Kaybı", "Loss", higher_is_better=False)

    # Bar grafik — test doğruluğu karşılaştırması
    if acc_aug is not None and acc_no is not None:
        ax_bar = fig.add_subplot(gs[2, :])
        bars = ax_bar.bar(
            ["Augmentation VAR", "Augmentation YOK"],
            [acc_aug * 100, acc_no * 100],
            color=[C_AUG, C_NOAUG], width=0.4,
            edgecolor="white", linewidth=1.5,
        )
        for bar, val in zip(bars, [acc_aug, acc_no]):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.05,
                        f"%{val * 100:.2f}", ha="center", va="bottom",
                        fontsize=13, fontweight="bold")
        ax_bar.set_ylim(min(acc_aug, acc_no) * 100 - 1, 100.5)
        ax_bar.set_ylabel("Test Doğruluğu (%)", fontsize=11)
        ax_bar.set_title("Test Seti Doğruluğu Karşılaştırması", fontsize=12, fontweight="bold")
        ax_bar.grid(True, axis="y", alpha=0.3)

        diff = (acc_aug - acc_no) * 100
        sign = "+" if diff >= 0 else ""
        ax_bar.set_xlabel(
            f"Augmentation farkı: {sign}{diff:.2f} puan",
            fontsize=10, labelpad=8,
        )

    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
