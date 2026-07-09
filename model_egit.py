# model_egit.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. SENARYO: Kaggle'daki mantığı simüle eden veri seti
# Özellikler: [Koku, Sapka_Rengi, Yasam_Alani]
# Koku: 0=Badem(Yenir), 1=Anason(Yenir), 2=Çürük(Zehirli), 3=Baharatlı(Zehirli) vb.
# Renk: 0=Kahve, 1=Sarı, 2=Beyaz
# Habitat: 0=Orman, 1=Çayır, 2=Atık

# Eğitim verisi (Örnek)
X = np.array([
    [0, 0, 0], [1, 1, 1], [0, 2, 0], # Yenilebilirler
    [2, 0, 2], [3, 1, 0], [2, 2, 2], # Zehirliler
    [0, 1, 1], [5, 0, 0], [2, 1, 1]  # Karışık
])

# Etiketler: 0 = Yenilebilir (Edible), 1 = Zehirli (Poisonous)
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 1])

# 2. MODEL: Random Forest (Raporundaki parametrelerle)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# 3. KAYDETME
joblib.dump(rf_model, 'mantar_model.pkl')
print("✅ Model başarıyla eğitildi ve 'mantar_model.pkl' olarak kaydedildi!")