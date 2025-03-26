# cold-start-emissions
Prediction of cold start emissions from internal combustion engines


## project strucure
```
ML_endtoend_test/
│
├── artifacts/                 # Saved models, datasets, reports
│
├── data/                      # Raw or external data
│   └── raw.csv
│
├── notebooks/                 # Jupyter notebooks for EDA, experimentation
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── logger.py              # Logging configuration
│   ├── exception.py           # Custom exception handling
│
│   ├── components/            # Data processing and model pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│
│   └── pipeline/              # Training and prediction pipelines
│       ├── __init__.py
│       ├── train_pipeline.py
│       ├── predict_pipeline.py
│
├── logs/                      # Logs generated during training/testing
│
├── tests/                     # Unit tests for components
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── main.py                    # Entry point for running training/prediction
```