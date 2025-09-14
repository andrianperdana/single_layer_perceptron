# Tugas - Single Layer Perceptron with Sigmoid Activation for Binary Classification on Iris Dataset
# Andrian Danar Perdana (23/513040/PA/21917)
# Andreandhiki Riyanta Putra (23/517511/PA/22191)
# Daffa Indra Wibowo (23/518514/PA/22253)
# Muhammad Argya Vityasy (23/522547/PA/22475)
# Rayhan Firdaus Ardian (23/519095/PA/22279)

# Impor library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Hyperparameter
LEARNING_RATE = 0.1
EPOCHS = 5  # Sheet berjalan selama 5 iterasi

# Data dari Sheet
# Data ditulis langsung di dalam kode (hardcode) agar sama persis dengan spreadsheet.
# Tidak ada proses pembagian data acak atau standardisasi.

# Data training (80 sampel)
X_train_data = np.array([
    [5.1, 3.5, 1.4, 0.2], [4.9, 3.0, 1.4, 0.2], [4.7, 3.2, 1.3, 0.2],
    [4.6, 3.1, 1.5, 0.2], [5.0, 3.6, 1.4, 0.2], [5.4, 3.9, 1.7, 0.2],
    [4.6, 3.4, 1.4, 0.2], [5.0, 3.4, 1.5, 0.2], [4.4, 2.9, 1.4, 0.2],
    [4.9, 3.1, 1.5, 0.2], [5.4, 3.7, 1.5, 0.2], [4.8, 3.4, 1.6, 0.2],
    [4.8, 3.0, 1.4, 0.2], [4.3, 3.0, 1.1, 0.2], [5.8, 4.0, 1.2, 0.2],
    [5.7, 4.4, 1.5, 0.2], [5.4, 3.9, 1.3, 0.2], [5.1, 3.5, 1.4, 0.2],
    [5.7, 3.8, 1.7, 0.2], [5.1, 3.8, 1.5, 0.2], [5.4, 3.4, 1.7, 0.2],
    [5.1, 3.7, 1.5, 0.2], [4.6, 3.6, 1.0, 0.2], [5.1, 3.3, 1.7, 0.2],
    [4.8, 3.4, 1.9, 0.2], [5.0, 3.0, 1.6, 0.2], [5.0, 3.4, 1.6, 0.2],
    [5.2, 3.5, 1.5, 0.2], [5.2, 3.4, 1.4, 0.2], [4.7, 3.2, 1.6, 0.2],
    [4.8, 3.1, 1.6, 0.2], [5.4, 3.4, 1.5, 0.2], [5.2, 4.1, 1.5, 0.2],
    [5.5, 4.2, 1.4, 0.2], [4.9, 3.1, 1.5, 0.2], [5.0, 3.2, 1.2, 0.2],
    [5.5, 3.5, 1.3, 0.2], [4.9, 3.1, 1.5, 0.2], [4.4, 3.0, 1.3, 0.2],
    [5.1, 3.4, 1.5, 0.2], [7.0, 3.2, 4.7, 1.4], [6.4, 3.2, 4.5, 1.5],
    [6.9, 3.1, 4.9, 1.5], [5.5, 2.3, 4.0, 1.3], [6.5, 2.8, 4.6, 1.5],
    [5.7, 2.8, 4.5, 1.3], [6.3, 3.3, 4.7, 1.6], [4.9, 2.4, 3.3, 1.0],
    [6.6, 2.9, 4.6, 1.3], [5.2, 2.7, 3.9, 1.4], [5.0, 2.0, 3.5, 1.0],
    [5.9, 3.0, 4.2, 1.5], [6.0, 2.2, 4.0, 1.0], [6.1, 2.9, 4.7, 1.4],
    [5.6, 2.9, 3.6, 1.3], [6.7, 3.1, 4.4, 1.4], [5.6, 3.0, 4.5, 1.5],
    [5.8, 2.7, 4.1, 1.0], [6.2, 2.2, 4.5, 1.5], [5.6, 2.5, 3.9, 1.1],
    [5.9, 3.2, 4.8, 1.8], [6.1, 2.8, 4.0, 1.3], [6.3, 2.5, 4.9, 1.5],
    [6.1, 2.8, 4.7, 1.2], [6.4, 2.9, 4.3, 1.3], [6.6, 3.0, 4.4, 1.4],
    [6.8, 2.8, 4.8, 1.4], [6.7, 3.0, 5.0, 1.7], [6.0, 2.9, 4.5, 1.5],
    [5.7, 2.6, 3.5, 1.0], [5.5, 2.4, 3.8, 1.1], [5.5, 2.4, 3.7, 1.0],
    [5.8, 2.7, 3.9, 1.2], [6.0, 2.7, 5.1, 1.6], [5.4, 3.0, 4.5, 1.5],
    [6.0, 3.4, 4.5, 1.6], [6.7, 3.1, 4.7, 1.5], [6.3, 2.3, 4.4, 1.3],
    [5.6, 3.0, 4.1, 1.3], [5.5, 2.5, 4.0, 1.3]
])

y_train_data = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

# Data validasi (20 sampel)
X_test_data = np.array([
    [5.0, 3.5, 1.3, 0.2], [4.5, 2.3, 1.3, 0.2], [4.4, 3.2, 1.3, 0.2],
    [5.0, 3.5, 1.6, 0.2], [5.1, 3.8, 1.9, 0.2], [4.8, 3.0, 1.4, 0.2],
    [5.1, 3.8, 1.6, 0.2], [4.6, 3.2, 1.4, 0.2], [5.3, 3.7, 1.5, 0.2],
    [5.0, 3.3, 1.4, 0.2], [5.5, 2.6, 4.4, 1.2], [6.1, 3.0, 4.6, 1.4],
    [5.8, 2.6, 4.0, 1.2], [5.0, 2.3, 3.3, 1.0], [5.6, 2.7, 4.2, 1.3],
    [5.7, 3.0, 4.2, 1.2], [5.7, 2.9, 4.2, 1.3], [6.2, 2.9, 4.3, 1.3],
    [5.1, 2.5, 3.0, 1.1], [5.7, 2.8, 4.1, 1.3]
])

y_test_data = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])


# Fungsi Sigmoid
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# Fungsi Loss Sum of Squared Error (SSE)
def sum_squared_error(target_aktual, prediksi):
    return ((target_aktual - prediksi) ** 2).mean()


# Single Layer Perceptron dengan Aktivasi Sigmoid
class SigmoidPerceptron:
    def __init__(self, jumlah_fitur, laju_pembelajaran=0.1):
        # Inisialisasi bobot dan bias ke 0.5 seperti di sheet
        self.bobot = np.full((jumlah_fitur,), 0.5)
        self.bias = 0.5
        self.laju_pembelajaran = laju_pembelajaran

    # Fungsi untuk memprediksi probabilitas
    def predict_proba(self, data_input):
        return sigmoid(np.dot(data_input, self.bobot) + self.bias)

    # Fungsi untuk memprediksi kelas biner (0 atau 1)
    def predict(self, data_input, threshold=0.5):
        return (self.predict_proba(data_input) >= threshold).astype(int)

    # Fungsi untuk melatih model
    def fit(self, data_latih, target_latih, epochs=100):
        # Dictionary untuk menyimpan riwayat loss dan akurasi
        riwayat_training = {"loss": [], "accuracy": []}

        # Looping utama untuk setiap epoch (iterasi di sheet)
        for epoch in range(epochs):
            # Looping untuk setiap sampel data (ini adalah logika Stochastic Gradient Descent)
            for sampel_input, target_aktual in zip(data_latih, target_latih):
                # Forward pass untuk satu sampel
                prediksi_probabilitas = self.predict_proba(sampel_input)

                # --- Backward Pass (Gradient Descent untuk satu sampel) ---
                # Logika ini sekarang cocok dengan aturan pembaruan di sheet yang menggunakan SSE
                error = prediksi_probabilitas - target_aktual
                turunan_sigmoid = prediksi_probabilitas * (1.0 - prediksi_probabilitas)
                
                # Perhitungan gradien untuk loss SSE
                gradient_bobot = 2 * error * turunan_sigmoid * sampel_input
                gradient_bias = 2 * error * turunan_sigmoid

                # Perbarui bobot dan bias langsung setelah setiap sampel
                self.bobot -= self.laju_pembelajaran * gradient_bobot
                self.bias -= self.laju_pembelajaran * gradient_bias

            # Setelah satu epoch selesai, hitung metrik untuk seluruh data training
            prediksi_satu_epoch = self.predict_proba(data_latih)
            loss = sum_squared_error(target_latih, prediksi_satu_epoch)
            akurasi = (self.predict(data_latih) == target_latih).mean()
            
            riwayat_training["loss"].append(loss)
            riwayat_training["accuracy"].append(akurasi)
            
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Akurasi: {akurasi:.4f}")

        return riwayat_training


# Inisialisasi model dengan parameter yang sesuai dengan sheet
model = SigmoidPerceptron(
    jumlah_fitur=X_train_data.shape[1], laju_pembelajaran=LEARNING_RATE
)

# Latih model
riwayat = model.fit(X_train_data, y_train_data, epochs=EPOCHS)

# Evaluasi model pada data pengujian
akurasi_testing = (model.predict(X_test_data) == y_test_data).mean()

print("\n--- Hasil Akhir ---")
print(f"Akurasi training terakhir: {riwayat['accuracy'][-1]:.4f}")
print(f"Akurasi testing: {akurasi_testing:.4f}")

# Membuat plot dari riwayat training
plt.figure(figsize=(12, 5))

# Plot Loss per epoch
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS + 1), riwayat["loss"])
plt.title("Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

# Plot Akurasi per epoch
plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS + 1), riwayat["accuracy"])
plt.title("Akurasi per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Akurasi")
plt.ylim(0, 1.1)  # Atur batas sumbu y agar visualisasi lebih baik
plt.grid(True)

plt.tight_layout()
plt.savefig("sheet_logic_plots.png")
plt.show()

