# E-Commerce Customer Churn — Prediksi Pelanggan Berisiko Churn

Notebook: `Customer_Churn_Project (2).ipynb`

Klasifikasi churn pelanggan e-commerce menggunakan **XGBoost** dengan pendekatan **CRISP-DM**. Fokus utama adalah **recall** karena biaya gagal mendeteksi churn (FN) jauh lebih besar daripada salah memberi kampanye retensi (FP).

## Ringkasan

| Aspek | Isi |
|---|---|
| Dataset | 5.630 pelanggan, 20 kolom ([Kaggle — Ankit Verma](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)) |
| Target | `Churn` (1 = churn, 0 = aktif), proporsi churn **16,8%** |
| Masalah utama | Imbalance kelas + 557 baris duplikat yang menyebabkan data leakage |
| Model final | XGBoost (`scale_pos_weight=5`) + `RandomizedSearchCV`, threshold **0,50** |
| Hasil test | **Recall 0,88 · Precision 0,71 · F1 0,79 · Accuracy 0,92 · ROC AUC 0,960** |

## Alur Kerja

1. **Business Understanding** — biaya FN = Rp 67.680, FP = Rp 14.700 (rasio ≈ 4,6:1) → recall jadi metrik utama, threshold dipilih dengan fungsi biaya `FN×67.680 + FP×14.700`.
2. **Pembersihan** — seragamkan ejaan kategori (`Mobile`/`Mobile Phone`, `Phone`), hapus **557 duplikat** sebelum split → 5.073 baris.
3. **Feature engineering** — `Tenure_Complain_Interaction = Tenure × Complain` dan `Cashback_per_Tenure = CashbackAmount / (Tenure + 1)`.
4. **Split 3 arah (stratified)**:
   - Train/Test = 80/20 (`random_state=20`)
   - Train → Fit (3.246) / Validation (812) (`random_state=42`)
   - Test (1.015) **dikunci**, dipakai sekali di akhir
5. **Preprocessing dalam pipeline** (bebas leakage): `KNNImputer` → `TargetEncoder` + `RobustScaler`. Tidak pakai SMOTE; imbalance ditangani via `scale_pos_weight=5`.
6. **Modeling** — bandingkan KNN, SVM, Logistic Regression, Random Forest, XGBoost dengan `StratifiedKFold` 5-fold yang sama; XGBoost terpilih. Tuning 50 kombinasi acak, `scoring='recall'`, hanya di data fit.
7. **Threshold** — grid 0,05–0,95 di data validasi → optimal **0,50** (kerugian validasi Rp 2.447.520), berimpitan dengan default.
8. **Evaluasi** — confusion matrix, ROC, learning curve dengan prediksi **OOF** (bukan in-sample), SHAP + feature importance.
9. **Deployment** — pipeline lengkap disimpan dengan `joblib`.

## Hasil (data test, 1.015 baris)

```text
[[787  60]   TN=787   FP=60
 [ 20 148]]  FN=20    TP=148
```

| Metrik | Nilai |
|---|---|
| Recall | 0,88 |
| Precision | 0,71 |
| F1-Score | 0,79 |
| Accuracy | 0,92 |
| ROC AUC | 0,960 |

**Dampak finansial:** FN 20 × Rp 67.680 + FP 60 × Rp 14.700 = **Rp 2.235.600**.

**Diagnosis overfitting:** train recall OOF = 0,8544 vs test recall = 0,8810 → gap **−0,0266** (tidak ada indikasi overfitting).

## Insight Utama (SHAP)

- `Cashback_per_Tenure` — paling berpengaruh; nilai tinggi mendorong ke churn.
- `Complain = 1` → churn; `Tenure` tinggi → cenderung aktif.
- `NumberOfAddress` dan `SatisfactionScore` juga berkontribusi.
- `Gender`, `HourSpendOnApp`, dan `Tenure_Complain_Interaction` kontribusinya marginal → kandidat penyederhanaan model.



