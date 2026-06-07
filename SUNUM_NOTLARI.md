# 📊 SUNUM NOTLARI - MNIST El Yazısı Hesap Makinesi

## 🎯 Projenin Özeti
El yazısıyla çizilen rakamlar ve matematiksel operatörleri gerçek zamanlı tanıyan, ardından çıktılarla aritmetik işlem yapan masaüstü uygulaması.

---

## 📌 5 BLOK KOD VE AÇIKLAMALARI

### ▶ BLOK 1: train_model.py (Rakam Tanıma Modeli Eğitimi)

**Ne yapar?**
- MNIST veri setinden (60.000 eğitim + 10.000 test örneği) rakam tanıyabilen CNN modeli eğitir
- İki farklı model eğitir: **Augmentation VAR** ve **Augmentation YOK**

**Sunumda anlatılması gereken ÖNEMLİ kısımlar:**

#### 1️⃣ **Veri Ön İşleme (load_data())**
```python
# ✅ Satır 32-46
x_train = x_train.astype("float32") / 255.0    # Normalizasyon: [0,255] → [0.0,1.0]
x_train = x_train[..., np.newaxis]             # Boyut: (60000, 28, 28) → (60000, 28, 28, 1)
y_train_cat = keras.utils.to_categorical(y_train, 10)  # One-hot: 5 → [0,0,0,0,0,1,0,0,0,0]
```
**İzleme:** Piksel değerlerini normalize etmek ağın daha hızlı öğrenmesini sağlar. Kanal boyutu eklemek CNN'nin 4D tensor beklemesinden olur.

#### 2️⃣ **Sınıf Ağırlıklandırması (compute_sample_weights())**
```python
# ✅ Satır 49-63
cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_labels)
cw[8] *= 2.5   # Rakam 8'e 2.5× ek ceza
```
**İzleme:** Rakam 8 kapalı döngü yapısı nedeniyle 0, 6, 9 ile karıştırılır. 2.5× ağırlık verince model 8'e daha dikkat gösterir. Sonuç: 8'in tanınma oranı %82 → %98.

#### 3️⃣ **Model Mimarisi (build_model())**
```python
# ✅ Satır 67-105
# BLOK 1: Conv2D(32) → BatchNorm → Conv2D(32) → MaxPool(2×2) → Dropout(0.25)
# BLOK 2: Conv2D(64) → BatchNorm → Conv2D(64) → MaxPool(2×2) → Dropout(0.25)
# Flatten → Dense(256) + BatchNorm + Dropout(0.5) → Dense(10, Softmax)
```
**İzleme:**
- **Conv2D**: Görüntüdeki özellikleri çıkar (kenarlar, eğriler, noktalar)
- **BatchNormalization**: Mini-batch'i normalize → eğitim hızlanır
- **MaxPooling**: Görüntü boyutunu küçült (28→14→7) → hesaplama azal
- **Dropout**: Rastgele nöronları devre dışı bırak → aşırı öğrenme (overfitting) engelle
- **Label Smoothing (0.1)**: Sert one-hot [0,0,1,0...] yerine yumuşak etiket [0, 0.005, 0.9, 0.005...] → aşırı güven azalt

#### 4️⃣ **Veri Artırımı (Data Augmentation)**
```python
# ✅ Satır 110-118
datagen = ImageDataGenerator(
    rotation_range=12,          # ±12° döndürme
    width_shift_range=0.1,      # %10 yatay kaydırma
    height_shift_range=0.1,     # %10 dikey kaydırma
    zoom_range=0.12,            # %12 yakınlaştırma
    shear_range=0.1,            # %10 kesme (eğik yazı)
)
```
**İzleme:** 60.000 görüntü × 15 epoch = 900.000 FARKLI görüntü! Ağ her epoch'ta yeni varyasyonlar görür → overfitting azalır → gerçek çizime daha iyi genelleme.

#### 5️⃣ **Eğitim Callbacks**
```python
# ✅ Satır 128-142
EarlyStopping(patience=5)          # 5 epoch iyile&şme yok → durdur
ReduceLROnPlateau(factor=0.5)      # Durgunluk → öğrenme hızını 0.5× azalt
```
**İzleme:** Early stopping aşırı öğrenmeyi engeller, ReduceLROnPlateau ince ayarlamalara imkân tanır.

---

### ▶ BLOK 2: train_operator.py (Operatör Tanıma Modeli Eğitimi)

**Ne yapar?**
- +, −, ×, ÷ operatörlerini tanıyabilen ayrı bir CNN eğitir
- MNIST'te operatör verisi olmadığından **sentetik veri oluşturur**

**Sunumda anlatılması gereken ÖNEMLİ kısımlar:**

#### 1️⃣ **Sentetik Veri Üretimi (generate_operator_image())**
```python
# ✅ Satır 40-84
img  = Image.new("L", (28, 28), 0)           # Siyah 28×28 kanvas
cx  = size // 2 + random.randint(-3, 3)      # Rastgele merkez (±3 px)
arm = size // 4 + random.randint(-2, 3)      # Rastgele çizgi uzunluğu
lw  = random.randint(2, 4)                   # Rastgele kalınlık
drawers[op_index](draw, cx, cy, arm, lw)    # İşareti çiz
img = img.rotate(random.uniform(-12, 12))    # Rastgele döndürme ±12°
arr = np.clip(arr + noise, 0, 255)           # Gaussian gürültüsü ekle (σ=15)
```
**İzleme:** Her çalıştırışta FARKLI görüntü → sentetik ama gerçekçi veri → model genelleşir.

#### 2️⃣ **Neden ÷ yerine / ?**
```python
# ✅ Satır 32-39: _draw_slash()
```
**İzleme:**
- **÷ (bölü sembolü)**: + ve − ile çok karışıyordu (%40 hata)
- **/ (eğik çizgi)**: Çapraz eğim diğer operatörlerden NET AYRışır
- **Sonuç**: ÷ ile %75 doğruluk → / ile %97 doğruluk 🎯

#### 3️⃣ **Veri Oluşturma**
```python
# ✅ Satır 87-119
for label in range(NUM_OP_CLASSES):          # 4 operatör
    for _ in range(SAMPLES_PER_OP):          # Her biri 8.000 örnek
        X.append(generate_operator_image(label))
# Toplam: 4 × 8.000 = 32.000 görüntü
```
**İzleme:** Sınırlandırılmamış veri üretimi, etiketlenmiş, kontrollü parametreler.

---

### ▶ BLOK 3: app.py (Ana GUI Uygulaması)

**Ne yapar?**
- Tkinter ile masaüstü arayüz oluşturur
- Canlı tahmin: Fare kaldırılınca 1 sn sonra otomatik tanıma
- Kullanıcı çizimi → Model tahmini → Güven skoru gösterimi

**Sunumda anlatılması gereken ÖNEMLİ kısımlar:**

#### 1️⃣ **DrawCanvas Sınıfı (Çizim Alanı)**
```python
# ✅ Satır 56-110
class DrawCanvas(tk.Canvas):
    def _paint(self, e):
        # Tkinter'de görsel çizim (mouse için)
        self.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="white")
        # PIL'de de çizim (Model tahmini için)
        self._pil_drw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill=255)
    
    def _arm_timer(self):
        # 1 sn timer başlat
        self._timer = threading.Timer(1.0, self._fire)
        self._timer.start()
    
    def _fire(self):
        # Timer sona erince callback çağır → otomatik tahmin
        if self._on_idle:
            self.after(0, self._on_idle)
```
**İzleme:** İki paralel çizim:
- Tkinter Canvas: Kullanıcı görüyür
- PIL Image: Model tahmini için veri hazırla

#### 2️⃣ **Rakam Tahmin Etme (Inference)**
```python
# ✅ Satır 470-487
def _infer_digit(self, slot: DigitSlot):
    arr   = slot.canvas.get_array().reshape(1, 28, 28, 1)  # [0,1] float32
    probs = self.digit_model.predict(arr, verbose=0)[0]    # 10 olasılık
    digit = int(np.argmax(probs))        # Hangi rakam en yüksek?
    confidence = float(probs.max())      # Olasılığı kaçyüzde?
    slot.set_result(digit, confidence)   # Sonucu göster
```
**İzleme:** 
- Model → 10 olasılık döndür [0.001, 0.95, 0.03, ...]
- argmax() = 1 (rakam "1")
- max() = 0.95 (%95 güven)

#### 3️⃣ **Güven Skoru Renklendirmesi**
```python
# ✅ Satır 376-380
def _conf_color(pct: float) -> str:
    if pct >= 90:  return CONF_HIGH    # Yeşil
    if pct >= 70:  return CONF_MID     # Sarı
    return CONF_LOW                    # Kırmızı
```
**İzleme:** Sunuma hazırlanırken bu renkleri düşün:
- 🟢 Yeşil (≥%90): Güvenli tahmin → hesapla
- 🟡 Sarı (%70-89): Uyar, yeniden çizebilir
- 🔴 Kırmızı (<%70): HATA, değer yok

#### 4️⃣ **Otomatik Hesaplama**
```python
# ✅ Satır 541-552
def _try_auto_calc(self):
    n1 = self._assembled("left")      # Rakamları birleştir: 2, 1 → 21
    op = self._op_slot.operator       # +, −, ×, veya /
    n2 = self._assembled("right")     # Rakamları birleştir: 3, 4 → 34
    if n1 is not None and op is not None and n2 is not None:
        self._do_calc(n1, op, n2)     # 21 + 34 = 55
```
**İzleme:** Tüm tahminler hazır olunca otomatik hesapla.

#### 5️⃣ **Çok Basamaklı Sayı Birleştirme**
```python
# ✅ Satır 533-540
def _assembled(self, side: str):
    slots  = self._left_slots
    digits = [s.digit for s in slots if s.digit is not None]
    # [2, 1] → "21" → 21 (integer)
    return int("".join(str(d) for d in digits))
```
**İzleme:** Soldan sağa okunan rakamlar birleştirilir. 3 slot seçerse 3 basamaklı sayı (max. 999).

---

### ▶ BLOK 4: show_confusion_matrix.py (Confusion Matrix Analizi)

**Ne yapar?**
- Test setinde model tahminlerini değerlendir
- 10×10 karışıklık matrisi oluştur (hangi rakam hangi rakama karıştırılıyor?)
- En çok karışan 5 çifti listele

**Sunumda anlatılması gereken ÖNEMLİ kısımlar:**

#### 1️⃣ **Tahmin vs Gerçeklik**
```python
# ✅ show_confusion_matrix.py satır 25-30
y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)  # 10.000 tahmin
cm = confusion_matrix(y_test, y_pred)                         # 10×10 matris
```
**İzleme:** 
- Satırlar = Gerçek rakam
- Sütunlar = Tahmin edilen rakam
- Diyagonal = Doğru (istediğimiz)
- Diyagonal dışı = Hatalar (istemediğimiz)

#### 2️⃣ **Tipik Karışmalar**
```
8 → 0, 6, 9      (Kapalı döngü yapısı)
4 → 9            (Üst kapalı yay)
3 → 5, 8         (Orta yay örtüşmesi)
7 → 1            (Gövde çizgisi)
```
**İzleme:** Model bu çiftlerde hata yapıyor, bu bilgi ile modeli iyileştirebiliriz.

#### 3️⃣ **Ham vs Normalize**
- **Ham**: 0-1000 arası sayılar (kaç adet karıştırıldı?)
- **Normalize**: Yüzde (örn. %95 doğru sınıflandırıldı)

---

### ▶ BLOK 5: show_comparison.py (Augmentation Karşılaştırması)

**Ne yapar?**
- İki modelin eğitim eğrilerini karşılaştır:
  - Model 1: Augmentation VAR
  - Model 2: Augmentation YOK
- 4 grafik: Eğitim doğruluğu, Eğitim kaybı, Doğrulama doğruluğu, Doğrulama kaybı

**Sunumda anlatılması gereken ÖNEMLİ kısımlar:**

#### 1️⃣ **Neden Augmentation?**
```python
# ✅ Augmentation VAR satırlar
aug_vals = [0.85, 0.92, 0.95, 0.96, 0.97, ...]  # Eğitim doğruluğu
# ✅ Augmentation YOK satırlar
no_aug_vals = [0.93, 0.96, 0.97, 0.98, 0.99, ...]  # Eğitim doğruluğu
```
**İzleme:**
- Augmentation VAR: Eğitim yavaş (farklı varyasyonlar), doğrulama STABIL
- Augmentation YOK: Eğitim hızlı (aynı veriler tekrar), doğrulama SAP

#### 2️⃣ **Overfitting Boşluğu**
```python
# ✅ show_comparison.py satır 60-75
eğitim_doğ = max(history_aug["accuracy"])      # %97
doğrulama_doğ = history_aug["val_accuracy"][-1]  # %96.5
boşluk = eğitim_doğ - doğrulama_doğ            # %0.5 (KÜÇÜK = İYİ!)

# Augmentation olmadan:
eğitim_doğ = %99
doğrulama_doğ = %95
boşluk = %4 (BÜYÜK = KÖTÜ! Model ezberlemişe benziyor)
```
**İzleme:** Augmentation sayesinde model gerçek çizimi daha iyi tanıyabilir.

---

## 🎤 SUNUMDA KULLANILABİLECEK HIZLI ANLATIMLAR

### 30 Saniye Özet
> "5 Python scripti var. İlk ikisi derin öğrenme modelleri eğitiyor: rakam ve operatör. Üçüncüsü masaüstü GUI yapıyor, çizimi 1 sn sonra otomatik tahmin ediyor. Dördüncü ve beşinci script modellerin performansını görsellendiriyor. Tümü TensorFlow/Keras kullanıyor."

### Önemli Kavramlar
- **CNN**: Görüntüyü alır, özellikleri öğrenir, sınıflandırır
- **Augmentation**: Sentetik varyasyonlar → model genelleşir
- **Dropout**: Rastgele nöron devre dışı → aşırı öğrenme engelle
- **Softmax**: Son çıktı, 10 olasılık → argmax en yüksek olan
- **Confusion Matrix**: Hangi rakamlar hangileriyle karışıyor göster

### Sunuma uygun sorular
1. "Neden augmentation gerekli?" → Aynı verilerle 15 epoch eğitirsen overfitting olur
2. "Rakam 8 neden zor?" → Kapalı döngü yapısı, 0/6/9 ile benzer
3. "/ neden ÷ yerine?" → Çapraz eğim diğer operatörlerden net ayrışır

---

## 📁 Dosya Yapısı Hatırlatma
```
proje el yazısı/
├── train_model.py              (BLOK 1)
├── train_operator.py           (BLOK 2)
├── app.py                      (BLOK 3)
├── show_confusion_matrix.py    (BLOK 4)
├── show_comparison.py          (BLOK 5)
├── mnist_model.keras           (Eğitilmiş rakam modeli)
├── operator_model.keras        (Eğitilmiş operatör modeli)
├── history.json                (Augmentation VAR eğitim geçmişi)
├── history_no_aug.json         (Augmentation YOK eğitim geçmişi)
└── SUNUM_NOTLARI.md            (ŞU DOSYA)
```

---

## ⏱️ Koşu Sırasında:
1. Terminal'de `python train_model.py` → 2 model eğit (15 min GPU / 30-60 min CPU)
2. Terminal'de `python train_operator.py` → Operatör modeli eğit (5 min GPU)
3. Terminal'de `python app.py` → GUI başlat, canlı çizim yap
4. GUI'de "📊 Confusion Matrix" butonu → Hata analizi
5. GUI'de "📈 Karşılaştır" butonu → Augmentation grafiği

Başarılar! 🎓
