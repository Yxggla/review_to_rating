# Data Layout

```text
data/
└── datasets/
    ├── original/
    │   ├── train.parquet
    │   ├── validation.parquet
    │   ├── test.parquet
    │   ├── train.csv
    │   ├── validation.csv
    │   └── test.csv
    └── processed_3class/
        ├── train_3class.csv
        ├── validation_3class.csv
        └── test_3class.csv
```

Notes:

- `original/` is the original Amazon Reviews Multi English split data.
- `processed_3class/` is the processed dataset with `label_3class`.
