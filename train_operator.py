"""
Operatör Tanıma Modeli Eğitimi
Sentetik veri ile +, −, ×, ÷ çizimlerini tanımayı öğrenir.
"""

import numpy as np
from PIL import Image, ImageDraw
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

OPERATORS       = ['+', '−', '×', '/']   # sınıf 0,1,2,3
OP_MODEL_PATH   = "operator_model.keras"
IMG_SIZE        = 28
NUM_OP_CLASSES  = 4
SAMPLES_PER_OP  = 8000
EPOCHS          = 30
BATCH_SIZE      = 128


# ─── Sentetik Görüntü Üretimi ─────────────────────────────────────────────────
def _draw_plus(draw, cx, cy, arm, lw):
    draw.line([cx - arm, cy, cx + arm, cy], fill=255, width=lw)
    draw.line([cx, cy - arm, cx, cy + arm], fill=255, width=lw)


def _draw_minus(draw, cx, cy, arm, lw):
    draw.line([cx - arm, cy, cx + arm, cy], fill=255, width=lw)


def _draw_multiply(draw, cx, cy, arm, lw):
    d = int(arm * 0.85)
    draw.line([cx - d, cy - d, cx + d, cy + d], fill=255, width=lw)
    draw.line([cx + d, cy - d, cx - d, cy + d], fill=255, width=lw)


def _draw_slash(draw, cx, cy, arm, lw):
    # Çapraz çizgi (/) — +, −, × ile karışmaz
    d = int(arm * 0.9)
    draw.line([cx - d, cy + d, cx + d, cy - d], fill=255, width=lw)


def generate_operator_image(op_index: int, size: int = IMG_SIZE) -> np.ndarray:
    """Tek bir operatör için rastgele el yazısı benzeri görüntü üret."""
    img  = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    cx  = size // 2 + random.randint(-3, 3)
    cy  = size // 2 + random.randint(-3, 3)
    arm = size // 4 + random.randint(-2, 3)
    lw  = random.randint(2, 4)

    drawers = [_draw_plus, _draw_minus, _draw_multiply, _draw_slash]
    drawers[op_index](draw, cx, cy, arm, lw)

    # Hafif döndürme
    img = img.rotate(random.uniform(-12, 12), fillcolor=0)

    # Gaussian-benzeri gürültü
    arr   = np.array(img, dtype="float32")
    noise = np.random.normal(0, 15, arr.shape)
    arr   = np.clip(arr + noise, 0, 255)

    return arr.astype("float32") / 255.0


def generate_dataset():
    X, y = [], []
    for label in range(NUM_OP_CLASSES):
        for _ in range(SAMPLES_PER_OP):
            X.append(generate_operator_image(label))
            y.append(label)
    X = np.array(X)[..., np.newaxis]   # (N, 28, 28, 1)
    y = np.array(y)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


# ─── Model Mimarisi ────────────────────────────────────────────────────────────
def build_operator_model():
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(NUM_OP_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ─── Ana ──────────────────────────────────────────────────────────────────────
def main():
    print("Sentetik operatör verisi üretiliyor...")
    X, y = generate_dataset()

    split   = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    print(f"Eğitim: {len(X_train)} | Doğrulama: {len(X_val)}")

    # Hafif augmentation üstüne ekle
    datagen = ImageDataGenerator(
        rotation_range=12,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )
    datagen.fit(X_train)

    model = build_operator_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nDoğrulama Doğruluğu: %{acc * 100:.2f}  |  Loss: {loss:.4f}")

    model.save(OP_MODEL_PATH)
    print(f"Operatör modeli kaydedildi: {OP_MODEL_PATH}")

    # Örnek tahminleri göster
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for label in range(NUM_OP_CLASSES):
        for j in range(2):
            img = generate_operator_image(label)
            pred = model.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 1), verbose=0)[0]
            axes[j, label].imshow(img.squeeze(), cmap="gray")
            axes[j, label].set_title(f"G:{OPERATORS[label]} T:{OPERATORS[np.argmax(pred)]}\n%{pred.max()*100:.0f}")
            axes[j, label].axis("off")
    plt.suptitle("Operatör Tahminleri (G=Gerçek, T=Tahmin)")
    plt.tight_layout()
    plt.savefig("operator_samples.png", dpi=120)
    plt.show()


if __name__ == "__main__":
    main()
