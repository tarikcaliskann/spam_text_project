# Türkçe SMS Spam Sınıflandırması

**Metin Ön-İşleme Adımlarının Etkisi**

Bu projede, Türkçe SMS mesajları üzerinde farklı metin ön-işleme adımlarının
spam sınıflandırma performansına etkisi incelenmiştir. Projede,
TF-IDF özellik çıkarımı ve Logistic Regression modeli kullanılmıştır.

---

## 📌 Proje Amacı

- Türkçe SMS mesajlarını **Spam** veya **Normal** olarak sınıflandırmak
- Metin ön-işleme adımlarının (küçük harfe çevirme, noktalama silme, sesli harf silme)
  model başarımına etkisini analiz etmek
- Kullanıcıların arayüz üzerinden bu adımları açıp kapatarak sonucu gözlemleyebilmesini sağlamak

---

## 📂 Kullanılan Veri Seti

Bu projede, açık kaynak olarak paylaşılan ve **:contentReference[oaicite:0]{index=0}** platformunda yayınlanan
**Türkçe SMS Spam Veri Seti** kullanılmıştır.

Veri seti içeriği:

- **Message**: SMS metni
- **GroupText**: Etiket bilgisi (Spam / Normal)

Veri seti, model eğitimi sırasında `train.py` dosyası içerisinde kullanılmıştır.

---

## 🧠 Kullanılan Yöntemler

### Metin Ön-İşleme

- Küçük harfe çevirme
- Noktalama işaretlerini silme
- Sesli harfleri silme (deneysel)

### Özellik Çıkarımı

- TF-IDF (Term Frequency – Inverse Document Frequency)

### Sınıflandırma Modeli

- Logistic Regression

---

## 📊 Model Değerlendirme

Model performansı aşağıdaki metrikler kullanılarak değerlendirilmiştir:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix (görselleştirme)

Ayrıca, farklı ön-işleme senaryoları karşılaştırılarak
ön-işleme adımlarının model başarımına etkisi grafiklerle analiz edilmiştir.

---

## 🖥️ Proje Yapısı

spam_text_project/
│
├── src/
│ ├── train.py # Model eğitimi, değerlendirme metrikleri ve grafikler
│ ├── app.py # Gradio arayüzü (modelin kullanımı)
│ └── transforms.py # Metin ön-işleme fonksiyonları
│
├── data/
│ └── sms.csv # Kaggle’dan alınan Türkçe SMS Spam veri seti
│
├── models/
│ ├── spam_model.joblib # Eğitilmiş Logistic Regression modeli
│ └── tfidf_vectorizer.joblib # TF-IDF vektörleştirici
│
└── README.md

## 📄 Dosya Açıklamaları

- **train.py**  
  Kaggle veri seti kullanılarak modelin eğitildiği dosyadır.  
  Metin ön-işleme, TF-IDF vektörleştirme, model eğitimi, değerlendirme metrikleri
  ve grafiksel analizler bu dosyada gerçekleştirilmiştir.

- **app.py**  
  Eğitilmiş model kullanılarak Gradio arayüzü üzerinden
  spam / normal SMS tahmini yapılmasını sağlar.
  Bu dosya veri setini doğrudan okumaz, yalnızca eğitilmiş modeli kullanır.

- **transforms.py**  
  Metin ön-işleme adımlarının modüler olarak tanımlandığı dosyadır.
  Küçük harfe çevirme, noktalama temizleme ve sesli harf silme gibi
  işlemler bu dosyada yer almaktadır.

## 🧠 Eğitim ve Kullanım Ayrımı

Bu projede model eğitimi ve model kullanımı birbirinden ayrılmıştır.

- **Model Eğitimi:** `train.py`
- **Model Kullanımı (Inference):** `app.py`

Bu yaklaşım, makine öğrenmesi projelerinde yaygın olarak kullanılan
akademik ve endüstriyel bir tasarım desenidir.

## 🎮 Gradio Demo

Gradio arayüzü sayesinde kullanıcılar:

- SMS metni girebilir
- Ön-işleme adımlarını (küçük harf, noktalama silme, sesli harf silme)
  açıp kapatabilir
- Modelin verdiği Spam / Normal tahminini anlık olarak gözlemleyebilir

Bu yapı, modelin davranışının kullanıcı tarafından
etkileşimli şekilde incelenmesini sağlamaktadır.
