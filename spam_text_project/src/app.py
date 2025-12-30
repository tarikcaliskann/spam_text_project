import gradio as gr
import joblib

from src.transforms import preprocess_text


# ===============================
# MODEL VE VECTORIZER YÜKLE
# ===============================
model = joblib.load("models/spam_model.joblib")
vectorizer = joblib.load("models/tfidf_vectorizer.joblib")


# ===============================
# SUNUM İÇİN HAZIR ÖRNEKLER
# ===============================
EXAMPLES = [
    # SPAM
    ["Tebrikler! 10.000 TL hediye kazandınız. Hemen linke tıklayın.", True, True, False],
    ["%50+%50 ODUL KAZANDIN!!", True, False, False],
    ["Son gün! %50 indirim fırsatı için şimdi tıklayın. SMS iptal: 1234", True, True, False],
    ["Ücretsiz hediye kazandınız. Bilgilerinizi almak için bağlantıya girin.", True, True, False],
    ["DIGITURK'TEN FIRSAT! SADECE BUGUNE OZEL ARAYIN 0212XXXXXXX", True, True, False],

    # NORMAL (HAM)
    ["Akşam biraz geç geliyorum, sen yemeğe başla.", True, True, False],
    ["Toplantı yarın saat 10’da, ona göre hazırlık yapalım.", True, True, False],
    ["Bugün dersten sonra kütüphaneye geçiyorum.", True, True, False],
    ["Tamam, haberleşiriz. İyi akşamlar.", True, True, False],
    ["İyiyim teşekkürler 😊 Sen nasılsın?", True, True, False],
]


# ===============================
# TAHMİN FONKSİYONU
# ===============================
def predict_spam(text, lowercase, remove_punc, remove_vowel):
    if text.strip() == "":
        return "Lütfen bir SMS metni giriniz."

    cleaned = preprocess_text(
        text,
        lowercase=lowercase,
        remove_punc=remove_punc,
        remove_vowel=remove_vowel
    )

    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    label = "🚨 SPAM" if prediction == "spam" else "✅ NORMAL"

    return f"""
Tahmin Sonucu: {label}

-------------------------
Temizlenmiş Metin:
{cleaned}
"""


# ===============================
# GRADIO ARAYÜZÜ
# ===============================
with gr.Blocks(title="Türkçe SMS Spam Sınıflandırması") as demo:

    gr.Markdown("""
    # 📩 Türkçe SMS Spam Sınıflandırması
    Bu uygulama, **Türkçe SMS mesajlarını** farklı ön-işleme adımlarından geçirerek
    **Spam** veya **Normal** olarak sınıflandırır.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            sms_input = gr.Textbox(
                label="📨 SMS Metni",
                lines=6,
                placeholder="Örnek: Tebrikler! Hemen ödülünüzü almak için tıklayın..."
            )

            gr.Markdown("### ⚙️ Ön-İşleme Ayarları")

            lowercase = gr.Checkbox(
                value=True,
                label="Küçük harfe çevir"
            )

            remove_punc = gr.Checkbox(
                value=True,
                label="Noktalama işaretlerini sil"
            )

            remove_vowel = gr.Checkbox(
                value=False,
                label="Sesli harfleri sil (deneysel)"
            )

            predict_btn = gr.Button("🔍 Tahmin Et", variant="primary")

        with gr.Column(scale=1):
            output = gr.Textbox(
                label="📊 Sonuç",
                lines=10
            )

    predict_btn.click(
        fn=predict_spam,
        inputs=[sms_input, lowercase, remove_punc, remove_vowel],
        outputs=output
    )

    # 🔥 SUNUM İÇİN HAZIR ÖRNEKLER
    gr.Examples(
        examples=EXAMPLES,
        inputs=[sms_input, lowercase, remove_punc, remove_vowel],
        label="📌 Hazır Örnekler (Sunum için tek tık)"
    )

    gr.Markdown("""
    ---
    **Model:** TF-IDF + Logistic Regression  
    **Amaç:** Ön-işleme adımlarının spam sınıflandırmaya etkisini incelemek
    """)


# ===============================
# HUGGING FACE İÇİN LAUNCH
# ===============================
if __name__ == "__main__":
    demo.launch()
