# Multi-label news classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/den-bene/multilabel-news-classification/blob/main/news_classification.ipynb)

Classification of news articles into 18 categories using TensorFlow. Each article has two or three labels. The dataset contains 11,000 articles, their publication year and month, and 10,000 Bag-of-Words features.

The network combines three inputs: text encoded by an Embedding and Bidirectional LSTM, Bag-of-Words counts, and standardized publication metadata. Two Dense layers combine these representations before 18 sigmoid outputs. Training uses binary cross-entropy, dropout, batch normalization and early stopping; four configurations compare network size, dropout and L2 regularization.

## Results

Articles are split into training, validation and test sets (7,705 / 1,647 / 1,648). Identical texts stay in the same split. The text vocabulary and metadata scaler are fitted on training data.

Model selection uses validation macro-F1 at threshold 0.5. Among configurations within 0.02 of the best score, the model with the smallest training/validation gap is selected; ties favor fewer parameters. The selected model is `stronger_dropout`, restored to epoch 5. Validation selects a global prediction threshold of 0.35.

| Test metric | Value |
| --- | ---: |
| Macro-F1 | 0.6951 |
| Micro-F1 | 0.7263 |
| Subset accuracy | 0.5631 |
| Macro precision | 0.7416 |
| Macro recall | 0.6661 |

Subset accuracy counts an article as correct only when all its labels match. The [notebook](news_classification.ipynb) includes the configuration comparison, results at threshold 0.5, and per-label metrics and F1 chart. Categories are identified by number because their names are not supplied.

## Run

Use Python 3.11 or 3.12. Install the dependencies:

```sh
python -m pip install -r requirements.txt
```

Place `input_data.pkl` in `data/` or beside the notebook, then open `news_classification.ipynb` and run all cells. In Colab, upload the dataset to the runtime's file panel. If packages are missing, upload `requirements.txt` and run `%pip install -r requirements.txt` in a cell.

The dataset is about 883 MB and is excluded from the repository. Its [original download location](https://frasca.di.unimi.it/MLDNN/input_data.zip) is provided for reference; availability has not been verified. Load pickle files only from a trusted source. The file contains `texts`, `years`, `months`, `X_bow` and `labels`.

Running the notebook creates `reports/` and `news_results.zip`, containing the trained model, preprocessing settings, split assignments, predictions and metrics. These files are excluded from Git. Download the archive before closing a Colab runtime.

## Limitations

Overfitting remains: the training/validation macro-F1 gap is 0.280 at threshold 0.5. Results use one random split and seed, with validation reused for checkpoint, configuration and threshold selection. The test is an internal evaluation of this collection, which has previously been used for development.

The supplied BoW vocabulary was extracted from the full collection, including held-out articles. A fully inductive evaluation would rebuild it from training texts. Identical-text grouping does not catch near-duplicates, and the random split does not measure performance on future years or another source. Month is treated as a numeric feature. No input-branch ablation was run, and the missing BoW vocabulary mapping prevents prediction from raw text alone.
