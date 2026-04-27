# Local Dataset Placement

This project expects the Amazon Reviews Multi English files to be placed here:

```text
data/amazon_reviews_multi_en/
├── summary.json
├── processed_3class/
│   ├── train_3class.csv
│   ├── validation_3class.csv
│   └── test_3class.csv
└── raw_parquet/
    ├── train.parquet
    ├── validation.parquet
    └── test.parquet
```

Large CSV, parquet, zip, model, and generated output files are intentionally ignored by git so the public repository stays lightweight.
