# cold-start-emissions
Prediction of cold start emissions from internal combustion engines


## project strucure
```
ML_endtoend_test/
│
├── data/                      # Raw or external data
│   └── raw.csv
│
├── notebooks/                 # Jupyter notebooks for EDA, experimentation
│   ├── encoder_decoder_final.py
│   ├── mlp_final.py
│
├── src/                       # Source code
│   ├── __init__.py
│   ├── model_architecture.py
│   ├── data_preperation.py
│   ├── training.py
│   ├── validation.py
│   ├── metrics_plotting.py
│
├── logs/                      # Logs generated during training/testing
│
├── models/                    # directory to save trained model checkpoints
│   ├── model_epoch_0.pt
│
├── requirements.txt           # Python dependencies
├── LICENSE                    # GPLv3 license
├── CITATION.md                # How to cite the software
├── authors.md                 # How to cite the software
└── README.md                  # Project overview
```