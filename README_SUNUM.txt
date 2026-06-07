# ✅ SUNUM HAZIRLIĞI - TAMAMLANDI

## 📋 Ne Yapıldı?

Sınıfta sunumunuz için 5 blok kodunuza **kapsamlı Türkçe açıklamalar** ekledim.

---

## 📁 Sunuma Hazırlanırken Okumanız Gereken Dosyalar

### 1️⃣ **SUNUM_NOTLARI.md** (ANA SUNUM BELGESI) ⭐⭐⭐
**BUNU OKUYUN!** Sunuma hazırlanırken esas belge.

İçeriği:
- ✅ Her blok kod için **ÖNEMLİ kısımlar** vurgulanmış
- ✅ **Kod satır numaraları** (nereden okuyacağınızı bilirsiniz)
- ✅ **Hızlı anlatımlar** (30 sn, 2 dakika özeti)
- ✅ **Sunumda kullanılacak sorular** (soru sorduğunda ne diyeceğiniz)
- ✅ **Koşu sırası**: Hangi komutu sırayla çalıştırırsınız

### 2️⃣ **INLINE_ACIKLAMALAR.txt** (KOD SAT›RLARI KILAVUZU)
Sunumda "şu satırı açıkla" dediğinde:
- Dosya adı
- Satır numarası
- Ne söyleyeceğiniz

Örnek:
```
Satır 32-46 — VERI ÖN İŞLEME:
✓ Piksel normalizasyonu: [0,255] → [0.0,1.0]
✓ Kanal boyutu eklemek: (28,28) → (28,28,1)
```

---

## 🎤 SUNUMDA NASIL KULLANACAĞINIZ?

### Senaryo 1: Hocaya Özel Kısım Sorarsa
**Hocanız:** "train_model.py'de Dropout ne yapıyor?"

**Siz:**
1. INLINE_ACIKLAMALAR.txt'i açın
2. "Dropout" arayın → Satır 128-142 bulun
3. SUNUM_NOTLARI.md'nin ilgili bölümü açın

### Senaryo 2: Akran Sunumda Kafası Karışırsa
**Arkadaş:** "Neden augmentation gerekli?"

**Siz:**
1. SUNUM_NOTLARI.md → "Veri Artırımı (Data Augmentation)" bölümünü açın
2. Hazır cevap orada: Overfitting engelleme, 900.000 varyasyon...

### Senaryo 3: Canlı Kod Çalıştırırken
**Sunumda "GUI'de tahmin yaptırmak isterse"**
1. SUNUM_NOTLARI.md → "⏱️ Koşu Sırasında" kısmını oku
2. Komutları sırayla çalıştır

---

## 📊 2 DOSYA KÜTÜPHANESI (Sunumda Açık Tutun)

### 📄 **SUNUM_NOTLARI.md** 
Masaüstüne yapıştırıp, sunumda açık tutun:
- Hızlı referans
- Hocaya verilen soru için cevap
- Kod bloğunun mantığı

### 📄 **INLINE_ACIKLAMALAR.txt**
"Şu satırı açıkla" dediğinde:
- Hangi dosya → Satırları → Ne diyeceğini bilirsiniz

---

## 💡 SUNUMDA VURGULANACAK 5 BLOK

### BLOK 1: train_model.py
**Açılış:** "MNIST veri setinden rakam tanıyabilen model eğitiyoruz"
- Veri ön işleme (normalizasyon + one-hot)
- Sınıf ağırlıklandırması (rakam 8'e özel)
- Model mimarisi (2 evrişim bloku)
- **Augmentation:** Veri artırımı ile overfitting engelleme

### BLOK 2: train_operator.py  
**Açılış:** "Operatör modeli sentetik veriyle eğitiyoruz"
- Sentetik veri üretimi (PIL ile çizim)
- **Neden / yerine ÷?** Çapraz eğim → net ayrışma
- Gaussian gürültüsü ve döndürme → gerçekçilik
- 32.000 görüntü (4 operatör × 8.000)

### BLOK 3: app.py
**Açılış:** "GUI'de kullanıcı çizimi okuyor, tahmin gösteriyor"
- DrawCanvas: Çizim alanı (160×160 → 28×28)
- Timer: 1 sn sonra otomatik tahmin
- **Inference:** Model tahmin, softmax olasılığı
- Çok basamaklı sayı: 2+1→21, 3+4→34, otomatik hesapla
- Güven skoru: Yeşil/Sarı/Kırmızı

### BLOK 4: show_confusion_matrix.py
**Açılış:** "Modelin hangi rakamları hangileriyle karıştırdığını gösteriyoruz"
- 10×10 matris (satır=gerçek, sütun=tahmin)
- Diyagonal: Doğru (%99+)
- Diyagonal dışı: Hatalar (rakam 8 sorun)
- Ham + Normalize görselleştirme

### BLOK 5: show_comparison.py
**Açılış:** "Augmentation'ın etkisini karşılaştırıyoruz"
- 2 model: Augmentation VAR vs YOK
- 4 grafik: Eğitim doğruluğu, kaybı, doğrulama doğruluğu, kaybı
- **Overfitting boşluğu:** Aug. VAR %0.5 vs Aug. YOK %4
- Sonuç: Augmentation overfitting'i engeller

---

## 🎯 SUNUMDAN 2 SAAT ÖNCE

1. **SUNUM_NOTLARI.md'yi okuyun** (Tamamını)
2. **INLINE_ACIKLAMALAR.txt'yi okuyun** (Hızlıca)
3. **Koşu sırasında hangi komutu ne zaman çalıştıracağınızı düşünün**
4. **Zorlayabileceği kısımları belirleyin** (Augmentation, confusion matrix logic)

---

## 📱 SUNUMDA AÇIK TUTACAĞINIZ ARAÇLAR

1. **VS Code** — Kodu okumak / açıklamak
2. **Terminal** — Komutları çalıştırmak
3. **Bu dosyaları açık tut:**
   - SUNUM_NOTLARI.md (Masaüstü)
   - INLINE_ACIKLAMALAR.txt (Masaüstü)
4. **Önemli:** GPU kullanıyorsanız eğitim öncesi model ağırlıklarını hazır bulundurun

---

## ⚠️ SÜRPRİZ SORULAR IÇIN CEVAP ANAHTARI

**"Neden augmentation gerekli?"**
→ SUNUM_NOTLARI.md, "Veri Artırımı" bölümü

**"Rakam 8 neden zor?"**
→ SUNUM_NOTLARI.md, "Sınıf Ağırlıklandırması" bölümü

**"Confusion matrix'te ne görüyoruz?"**
→ SUNUM_NOTLARI.md, "BLOK 4" bölümü

**"Overfitting nedir, nasıl engellendi?"**
→ SUNUM_NOTLARI.md, "BLOK 5" bölümü

---

**Başarılar! 🎓 Sorularınız varsa bu dosyaları oku, cevap orada var.** 🚀
