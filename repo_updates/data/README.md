# Data

This project uses the real **NeurIPS 2022 Bank Account Fraud** benchmark
(`BaseDataset.csv`: 1,000,000 applications, 31 features + `fraud_bool` target,
~1.1% fraud). No synthetic or mock data is generated anywhere in this repo.

Place the file here:

    data/BaseDataset.csv

Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

The file (~213 MB) is git-ignored because GitHub rejects files over 100 MB.
To keep it elsewhere, use `--data /path/BaseDataset.csv` or set the
`BASE_DATASET_PATH` environment variable.
