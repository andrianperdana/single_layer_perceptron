# Tugas - Single Layer Perceptron with Sigmoid Activation for Binary Classification on Iris Dataset

Proyek ini adalah implementasi dari *Single Layer Perceptron* dengan fungsi aktivasi sigmoid untuk melakukan klasifikasi biner pada Iris dataset, yang direplikasi berdasarkan logika dari pengerjaan di spreadsheet.

## 👨‍💻 Anggota Kelompok

* **Andrian Danar Perdana** (23/513040/PA/21917)
* **Andreandhiki Riyanta Putra** (23/517511/PA/22191)
* **Daffa Indra Wibowo** (23/518514/PA/22253)
* **Muhammad Argya Vityasy** (23/522547/PA/22475)
* **Rayhan Firdaus Ardian** (23/519095/PA/22279)

## 🚀 Cara Menjalankan Kode

1.  Pastikan semua *library* yang dibutuhkan sudah ter-install dengan menjalankan perintah berikut:
    ```bash
    pip install -r requirements.txt
    ```
2.  Jalankan file `perceptron2.py` untuk melatih model berdasarkan logika spreadsheet:
    ```bash
    python perceptron2.py
    ```
3.  Skrip akan melatih model selama 5 epoch, menampilkan hasil akurasi di *console*, dan menyimpan grafik `sheet_logic_plots.png`.

## 📊 Hasil

Model dilatih selama **5 epoch** dengan *learning rate* **0.1** menggunakan *Sum of Squared Error* sebagai fungsi *loss*. Hasil yang didapatkan adalah sebagai berikut:

* **Akurasi training terakhir**: 0.9000
* **Akurasi testing**: 0.9500

### Grafik Pelatihan

Berikut adalah grafik *loss* dan akurasi selama proses pelatihan yang dihasilkan oleh `perceptron2.py`.

![sheet_logic_plots.png](sheet_logic_plots.png)
