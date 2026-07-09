from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Modeli yükle
try:
    model = joblib.load('mantar_model.pkl')
except:
    print("HATA: Önce model_egit.py dosyasını çalıştırmalısın!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Formdan verileri al
        try:
            koku = int(request.form['koku'])
            renk = int(request.form['renk'])
            habitat = int(request.form['habitat'])
            
            # Tahmin için veriyi hazırla
            veri = np.array([[koku, renk, habitat]])
            
            # Modele sor
            tahmin = model.predict(veri)[0]
            
            # Sonucu yorumla
            if tahmin == 0:
                sonuc_mesaji = "YENİLEBİLİR 🍄"
                css_class = "safe"
                detay = "Bu mantar özellikleri bakımından güvenli görünüyor."
            else:
                sonuc_mesaji = "DİKKAT! ZEHİRLİ OLABİLİR ☠️"
                css_class = "danger"
                detay = "Bu mantar yüksek risk taşıyor, kesinlikle tüketilmemeli."
                
            return render_template('index.html', prediction=sonuc_mesaji, detail=detay, css_class=css_class)
            
        except Exception as e:
            return render_template('index.html', error="Bir hata oluştu. Lütfen seçimleri kontrol edin.")

if __name__ == '__main__':
    app.run(debug=True)