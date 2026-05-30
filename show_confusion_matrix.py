"""
Confusion Matrix görüntüleme scripti (app.py'den çağrılır)
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report
import os

MODEL_PATH = "mnist_model.keras"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: {MODEL_PATH} bulunamadı. Önce train_model.py çalıştırın.")
        return

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = x_test[..., np.newaxis]

    model = keras.models.load_model(MODEL_PATH)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # ── Görselleştirme ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — MNIST El Yazısı Rakam Tanıma", fontsize=15, fontweight="bold")

    # Ham sayılar
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[0], linewidths=0.5, annot_kws={"size": 9})
    axes[0].set_xlabel("Tahmin Edilen Rakam", fontsize=12)
    axes[0].set_ylabel("Gerçek Rakam", fontsize=12)
    axes[0].set_title("Ham Sayılar", fontsize=13)

    # Normalize
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    mask_diag = np.eye(10, dtype=bool)
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="YlOrRd",
                xticklabels=range(10), yticklabels=range(10),
                ax=axes[1], linewidths=0.5, annot_kws={"size": 8},
                vmin=0, vmax=1)
    axes[1].set_xlabel("Tahmin Edilen Rakam", fontsize=12)
    axes[1].set_ylabel("Gerçek Rakam", fontsize=12)
    axes[1].set_title("Normalize (Oransal)", fontsize=13)

    # En çok karışan çiftleri bul
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)
    top5 = []
    flat = cm_no_diag.flatten()
    top_idx = np.argsort(flat)[::-1][:5]
    for idx in top_idx:
        r, c = divmod(idx, 10)
        top5.append((r, c, cm_no_diag[r, c]))

    info_text = "En Çok Karışan Çiftler:\n" + "\n".join(
        f"  {r} → {c}: {cnt} kez" for r, c, cnt in top5
    )
    fig.text(0.5, 0.01, info_text, ha="center", fontsize=10,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred, target_names=[str(i) for i in range(10)]))


if __name__ == "__main__":
    main()
