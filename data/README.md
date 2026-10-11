# Data

File `raw/credit_card_clients.xls` adalah salinan byte-identik file asli UCI.
Pembuat: I-Cheng Yeh. Citation: Yeh, I. (2009). Default of Credit Card Clients
[Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.
Sumber: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients.
Lisensi data: Creative Commons Attribution 4.0 International (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. Lisensi kode MIT tidak mengganti
lisensi data. Metadata unduhan/checksum ada di `raw/provenance.json`.

30.000 pemegang kartu kredit Taiwan; histori April-September 2005.
Target `default payment next month`: 1 = default; 0 = non-default, bulan berikutnya.
Definisi target mengikuti dataset; jangan mengasumsikan horizon 12 bulan atau
definisi 90 days past due. Nominal uang: New Taiwan Dollar (NTD), bukan rupiah.

| Input model | Makna |
| --- | --- |
| LIMIT_BAL | Limit kredit dalam NTD |
| PAY_0, PAY_2, ..., PAY_6 | Status pembayaran dari bulan terbaru ke terlama |
| BILL_AMT1, ..., BILL_AMT6 | Tagihan enam bulan, terbaru ke terlama, NTD |
| PAY_AMT1, ..., PAY_AMT6 | Pembayaran enam bulan, terbaru ke terlama, NTD |

Urutan historis: September, Agustus, Juli, Juni, Mei, April 2005.
Perhatikan bahwa PAY_0 adalah bulan pertama dan tidak ada PAY_1.
Kode resmi UCI: -1 = pay duly; 1-8 = terlambat 1-8 bulan; 9 = 9+ bulan.
Kode -2 dan 0 muncul di file, tetapi tidak didefinisikan rinci pada dokumentasi
publik UCI. Model mempertahankannya sebagai kategori, tanpa menebak maknanya.
Negatif pada BILL_AMT diizinkan, karena ada saldo kredit di data asli.

ID, SEX, EDUCATION, MARRIAGE, AGE tidak dipakai sebagai fitur prediksi.
Penghapusan demografi tidak cukup untuk menyimpulkan fairness.
Data tidak memiliki income, loan_amount, atau employment_length.

`examples/applicant.json` adalah contoh input simulasi yang dibuat untuk demo.
Contoh ini tidak memiliki label default aktual dan bukan tambahan data pelatihan.
`examples/high_delay.json` memakai nilai keuangan sama dengan status terlambat.
Keduanya hanya membantu memeriksa alur input-output aplikasi.
