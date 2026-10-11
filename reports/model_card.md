# Model card - Credit Scoring v1

## Tujuan dan populasi
Estimasi probabilitas label default pembayaran bulan berikutnya untuk pemegang
kartu kredit pada dataset historis Taiwan 2005. Ini proyek portofolio/pembelajaran,
bukan estimasi risiko kredit konsumen Indonesia atau model underwriting pinjaman baru.
Target mengikuti label UCI, bukan definisi buatan '90 hari terlambat' atau PD 12 bulan.
Semua nilai uang memakai NTD.

## Data dan hak penggunaan
I-Cheng Yeh (2009), Default of Credit Card Clients, UCI Machine Learning Repository.
DOI: https://doi.org/10.24432/C55S3H. Lisensi data: CC BY 4.0.
30.000 baris asal, 23 fitur asal, April-September 2005; target bulan berikutnya.
File mentah disertakan tanpa perubahan. SHA-256: `30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933`.
19 fitur pembayaran/keuangan dipakai. ID, SEX, EDUCATION, MARRIAGE, dan AGE dikeluarkan.
Mengeluarkan fitur demografis tidak membuktikan model adil; proxy bias masih mungkin.
Data hanya pelanggan yang sudah memperoleh kredit sehingga selection bias tetap ada.

## Pembagian dan pencegahan kebocoran
Seed 42. Sekitar 60% train, 15% calibration, 10% validation, 15% test.
Profil dengan 19 fitur identik, termasuk yang targetnya berbeda, berada di satu split.
Jumlah grup yang menyeberangi split: 0. Rasio aktual mengikuti ukuran grup.
Imputer/scaler/koefisien hanya fit pada train. Kalibrasi sigmoid hanya fit pada
calibration. Pemilihan kandidat dan cutoff hanya memakai validation.
Test digunakan setelah pilihan model dan cutoff dibekukan.
Split acak per profil digunakan karena satu kohort ini tidak memungkinkan
pengujian lintas waktu. Ini tidak membuktikan kinerja pada data masa depan.

| Split | Baris | Proporsi default |
| --- | ---: | ---: |
| train | 17,937 | 22.05% |
| calibration | 4,562 | 22.10% |
| validation | 3,053 | 22.67% |
| test | 4,448 | 22.05% |

## Model dan fitur
Logistic Regression, C=1, regularisasi L2, lbfgs. Kandidat: class_weight=None
dan balanced, keduanya dikalibrasi. Kandidat dipilih berdasarkan log loss validation
terendah (cost validation menjadi tie-breaker). Model terpilih: `logistic_standard`.
Nominal uang memakai signed log1p; fitur numerik distandardisasi. Status pembayaran
dikodekan one-hot -2 sampai 9. Kode -2 dan 0 dipertahankan sebagai kategori asli;
deskripsi publik UCI tidak mendefinisikan keduanya secara rinci.
Fitur tambahan: rasio tagihan/limit, proxy pembayaran/tagihan, perubahan tagihan,
jumlah bulan terlambat, dan keterlambatan terpanjang. Proxy bukan coverage cicilan.
Rasio ekstrem dibatasi secara deterministik di features.py; tidak belajar dari test.

## Hasil test
| Metrik | Logistic terkalibrasi | Baseline probabilitas konstan |
| --- | ---: | ---: |
| ROC-AUC | 0.7609 | 0.5000 |
| PR-AUC / Average Precision | 0.5203 | 0.2205 |
| Brier score (lebih kecil lebih baik) | 0.1393 | 0.1719 |
| Log loss (lebih kecil lebih baik) | 0.4427 | 0.5276 |
| Recall default | 0.7615 | 1.0000 |
| Precision default | 0.3527 | 0.2205 |

Cutoff flag review: 0.14, dipilih dengan grid 0.01-0.99 untuk
meminimalkan cost ilustratif FN:5, FP:1 pada validation.
Bobot ini bukan estimasi kerugian bank. Tidak memasukkan LGD, EAD, pendapatan,
kapasitas review, atau biaya operasional. Flag bukan keputusan approve/reject.
Confusion matrix test, baris aktual 0/1 dan kolom prediksi 0/1:
`[[2096, 1371], [234, 747]]`. Accuracy: 0.6392, KS: 0.4053.

## PD, skor, dan penjelasan
PD berasal dari sigmoid(slope * logit_model + intercept) pada kalibrasi terpisah.
Tidak ada jaminan PD terkalibrasi untuk populasi lain atau tahun lain.
Skor demo = 600 + 50/ln(2) * ln(((1-PD)/PD)/50), ditampilkan pada rentang 300-850.
Skor 600 berarti good:bad odds 50:1. Ini skala proyek sendiri, bukan FICO/SLIK.
Band demo: LOW <10%, MEDIUM 10%-<30%, HIGH >=30%. Band tidak sama dengan cutoff review.
Penjelasan adalah kontribusi linear pada log-odds terkalibrasi relatif rata-rata
fitur transformasi train. Seluruh kontribusi + intercept merekonstruksi PD tepat.
Penjelasan menunjukkan asosiasi model, bukan sebab, rekomendasi perilaku, atau
adverse-action reason codes yang tervalidasi untuk produksi.

## Batas dan pengembangan
Belum divalidasi dengan data bank Indonesia, split lintas waktu, audit fairness,
reject inference, monitoring drift, atau validasi independen. Tidak ada fitur
income, loan_amount, employment_length: dataset ini memang tidak memuatnya.
Untuk underwriting pinjaman baru, pilih data application dengan outcome/horizon jelas.
Jangan gunakan pendapatan fiktif untuk mengisi fitur yang tidak tersedia.
Joblib hanya dimuat dari artefak proyek yang dipercaya; jangan unggah model asing.
Versi model: `uci-lr-v1-30c6be3abd`. scikit-learn: 1.8.0.

## Referensi paper
Dastile, X., Celik, T., & Potsane, M. (2020). Statistical and machine learning models
in credit scoring: A systematic literature survey. Applied Soft Computing, 91, 106263.
https://doi.org/10.1016/j.asoc.2020.106263.
Proyek menerapkan baseline transparan, evaluasi imbalance, dan transparansi yang
dibahas survey; proyek ini bukan reproduksi seluruh eksperimen dalam paper.
