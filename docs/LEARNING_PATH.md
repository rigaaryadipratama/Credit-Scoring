# Urutan memahami proyek

## Tahap 1 - Jalankan dan pahami output

Buka dashboard dan bandingkan dua skenario. Jawab sendiri:
apa arti PD 6,81%, mengapa band berbeda dari cutoff review, dan mengapa skor
lebih tinggi berarti risiko lebih rendah? PD adalah estimasi peluang suatu label
default pada horizon/populasi yang ditentukan, bukan kepastian individu akan default.

## Tahap 2 - Pahami problem dan data

Baca data/README.md, lalu notebook eksplorasi jika ingin.
Sebutkan unit NTD, urutan bulan, target next month, dan alasan income tidak dipakai.
Bedakan application scoring pada pemohon baru dengan behavioral scoring pelanggan lama.
Jangan mengganti judul menjadi loan default underwriting tanpa mengganti data/model.

## Tahap 3 - Ikuti kode model

Baca berurutan: data.py, features.py, train.py, evaluation.py, scoring.py.
Pelajari train/calibration/validation/test, sigmoid, koefisien, standardisasi,
class imbalance, dan kontribusi pada log-odds. Tidak perlu memulai dengan neural network.

## Tahap 4 - Nilai kinerja dengan benar

Bandingkan ROC-AUC, Average Precision, Brier, dan recall/precision.
Accuracy saja tidak cukup pada kelas yang tidak seimbang.
Cutoff rendah biasanya meningkatkan recall dan volume review, sekaligus false positive.
Ganti cost ilustratif hanya sesudah memahami bahwa FN=5 dan FP=1 bukan angka bisnis nyata.
Evaluasi baru membutuhkan holdout baru jika keputusan berulang dibuat dari hasil test lama.

## Tahap 5 - Kembangkan setelah baseline dipahami

| Pengembangan | Alasan |
| --- | --- |
| Dataset pinjaman application | Agar fitur income/loan_amount dan outcome sesuai proyek pinjaman baru |
| Random Forest atau gradient boosting sebagai challenger | Bandingkan ranking, kalibrasi, penjelasan, dan cost pada protokol evaluasi sama |
| WoE/IV dan scorecard binning | Pelajari pendekatan scorecard dengan fit bin hanya pada train |
| Split lintas waktu pada data yang mendukung | Uji kemampuan generalisasi setelah waktu berubah |
| Audit fairness | Periksa metrik dan kalibrasi per kelompok; penghapusan kolom tidak cukup |
| Monitoring distribusi fitur dan outcome | Ketahui kapan populasi/pola pembayaran berubah |
| Estimasi expected loss dengan EAD/LGD yang valid | Hubungkan PD dengan kerugian, setelah datanya tersedia |

API dapat ditambahkan bila aplikasi lain perlu mengonsumsi prediksi. Untuk proyek
pertama ini, dashboard langsung ke model sudah cukup.

## Cerita proyek saat interview

"Saya membangun baseline transparan untuk default kartu kredit pada data publik UCI.
Saya memisahkan train, kalibrasi, validasi, dan test, serta menjaga profil identik
tetap dalam satu split. Saya mengevaluasi ranking dan kualitas probabilitas, bukan
hanya accuracy. Model disertai penjelasan yang merekonstruksi PD serta batas
penggunaan yang eksplisit. Dashboard memanggil pipeline yang sama dengan training."

Tambahkan angka aktual dari reports/metrics.json. Jangan mengklaim proyek dipakai
bank, production-ready, atau memprediksi nasabah Indonesia tanpa bukti.
