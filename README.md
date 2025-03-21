# 📌 Alur Pembuatan Machine Learning untuk Aplikasi Pencatatan Gula Darah & Hipertensi

Aplikasi ini menggunakan Machine Learning untuk **memprediksi risiko kesehatan** dan memberikan **rekomendasi makanan sehat** berdasarkan pencatatan gula darah dan tekanan darah pengguna. Model yang digunakan terdiri dari **Classification (Supervised Learning)** dan **Clustering (Unsupervised Learning)**.

---

## 🚀 Tahapan Pembuatan Machine Learning

### **1️⃣ Pengumpulan & Persiapan Data**

1. **Data yang Diperlukan**:
   - **Data Demografi** → Usia, Jenis Kelamin, Berat Badan, Tinggi Badan.
   - **Riwayat Kesehatan** → Diabetes, Hipertensi, Kolesterol.
   - **Gula Darah** → Puasa & setelah makan.
   - **Tekanan Darah** → Sistolik & Diastolik.
   - **Gaya Hidup** → Aktivitas fisik, Pola makan, Kebiasaan merokok.
   - **Catatan Konsumsi Makanan** → Karbohidrat, gula, serat, protein, dll.

2. **Preprocessing Data**:
   - Membersihkan data (menghapus data kosong/duplikat).
   - Normalisasi data (standarisasi nilai dengan skala 0-1 atau Z-score).
   - Encoding kategori (misal, jenis kelamin laki-laki/perempuan menjadi 0/1).
   - Feature selection untuk memilih fitur yang paling relevan.

---

### **2️⃣ Model Machine Learning**

#### **📍 Classification Model (Prediksi Risiko Kesehatan)**
- **Tujuan:** Menentukan apakah pengguna masuk kategori risiko **rendah, sedang, atau tinggi**.
- **Algoritma yang Digunakan:**
  - Logistic Regression → Model dasar untuk klasifikasi.
  - Random Forest → Model lebih kompleks & akurat.
  - XGBoost → Cocok untuk dataset besar.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
```

#### **📍 Clustering Model (Rekomendasi Makanan Sehat)**
- **Tujuan:** Mengelompokkan pengguna berdasarkan pola kesehatan mereka.
- **Algoritma yang Digunakan:**
  - K-Means Clustering → Mengelompokkan berdasarkan pola kesehatan.
  - DBSCAN → Mendeteksi pengguna dengan pola ekstrim (anomali).

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Tentukan jumlah cluster (misal, 3 kelompok)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualisasi hasil clustering
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='viridis')
plt.xlabel('Gula Darah')
plt.ylabel('Tekanan Darah')
plt.title('Clustering Pasien Berdasarkan Kesehatan')
plt.show()
```

| Cluster | Karakteristik | Rekomendasi Makanan |
|---------|--------------|---------------------|
| **Cluster 1** | Gula darah normal, tekanan darah stabil | Diet seimbang, konsumsi karbohidrat kompleks |
| **Cluster 2** | Gula darah tinggi, hipertensi ringan | Diet rendah gula & garam, konsumsi serat lebih banyak |
| **Cluster 3** | Diabetes parah, hipertensi berat | Diet ketat tanpa gula & garam, konsumsi protein tinggi |

---

### **3️⃣ Integrasi Model ke Aplikasi**

Jika ingin menghubungkan model dengan aplikasi berbasis web/mobile, gunakan API:

```python
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load Model yang sudah dilatih
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict/")
async def predict(data: dict):
    input_data = np.array([data['usia'], data['gula_darah'], data['tekanan_darah']]).reshape(1, -1)
    result = model.predict(input_data)
    return {"kategori_risiko": result[0]}
```

---

### **4️⃣ Evaluasi Model**

- **Evaluasi Classification** → Gunakan **Accuracy, Precision, Recall, F1-score**.
- **Evaluasi Clustering** → Gunakan **Silhouette Score atau Davies-Bouldin Index**.
- **Uji coba aplikasi dengan data nyata** untuk memastikan rekomendasi sesuai.

---

### **5️⃣ Deployment & Penggunaan Aplikasi**

- **Jika berbasis Web:** Model bisa di-deploy ke **Google Cloud, AWS, atau Heroku**.
- **Jika berbasis Mobile:** Model bisa dikonversi ke **TF Lite (untuk Android) atau CoreML (untuk iOS)**.

---

## 🎯 **Kesimpulan**
✅ **Classification (Supervised Learning)** → Prediksi kategori risiko kesehatan.
✅ **Clustering (Unsupervised Learning)** → Kelompokkan pengguna untuk rekomendasi makanan personal.
✅ **Integrasi ke API/Web/App** → Mempermudah pengguna dalam pencatatan dan mendapatkan saran makanan.

---

💡 **Jika ingin berkontribusi atau memiliki pertanyaan, silakan buat issue atau pull request! 🚀**
