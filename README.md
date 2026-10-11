# Multi-label news classification

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/den-bene/multilabel-news-classification/blob/main/news_classification.ipynb)

A deep-learning project predicting 18 semantic labels from news text, Bag-of-Words counts and publication metadata. Each article has two or three labels. The analysis covers preprocessing, a three-input neural network, regularization, validation-based model selection and held-out evaluation.

## Data and architecture

The dataset contains 11,000 news entries, 10,000 Bag-of-Words features, publication year/month and a binary target matrix with 18 columns. The dataset file is excluded from this repository; see [DATASET.md](DATASET.md).

The text branch uses a training-only vocabulary, separate unknown/padding IDs, a masked Embedding and Bidirectional LSTM. Two Dense branches encode BoW counts and standardized year/month. The representations are concatenated and passed through two Dense blocks with batch normalization and dropout, comparing L2 penalties on the BoW Dense kernel, followed by 18 sigmoid outputs. Binary cross-entropy trains the multi-label predictor. The notebook includes the architecture diagram and tensor shapes.

## Evaluation procedure

Use seed 213 and grouped random splits targeting 70% training, 15% validation and 15% test. Identical texts after lowercasing and whitespace normalization stay together. Actual split counts are 7,705, 1,647 and 1,648; 19 entries repeat an existing text. Training fits the vocabulary and metadata scaler. The supplied BoW representation is retained.

Compare four configurations of the same architecture for up to ten epochs each. Early stopping restores the checkpoint with the lowest validation objective (binary cross-entropy plus the L2 penalty). Among configurations within 0.02 validation macro-F1 of the highest score at threshold 0.5, select the smallest absolute train/validation gap, breaking ties by fewer parameters. This rule allows a modest performance tradeoff to control fitting capacity and avoids selecting solely by gap.

The selected checkpoint is reused without retraining. Select a single global threshold on validation macro-F1 from 0.20 to 0.70 in steps of 0.05. Freeze all choices before test predictions. Report the fixed 0.5 reference too, without choosing from test outcomes.

## Measured results

At threshold 0.5 during validation selection:

| Configuration | BoW L2 | Training macro-F1 | Validation macro-F1 | Absolute gap | Best epoch |
| --- | ---: | ---: | ---: | ---: | ---: |
| small_regularized | 0.0001 | 0.9827 | 0.6821 | 0.3007 | 4 |
| base | 0 | 0.9805 | 0.6708 | 0.3097 | 4 |
| stronger_dropout | 0 | 0.9606 | 0.6806 | 0.2800 | 5 |
| dropout_with_l2 | 0.001 | 0.9314 | 0.6488 | 0.2826 | 5 |

Selected configuration: **stronger_dropout** (BoW L2 coefficient 0), checkpoint epoch **5**, validation-selected threshold **0.35**. The training/validation macro-F1 gap remains **0.280** at threshold 0.5; overfitting is still substantial.

Final internal test results:

| Metric | Value |
| --- | ---: |
| Macro-F1 | 0.6951 |
| Micro-F1 | 0.7263 |
| Subset accuracy | 0.5631 |
| Macro precision | 0.7416 |
| Macro recall | 0.6661 |

Subset accuracy requires all labels of an article to be correct. The notebook contains threshold comparison, per-label precision/recall/F1 and confusion counts. Label names are not supplied, so reports use IDs.


The per-label F1 chart is included in the executed notebook.

The complete notebook ran successfully in a fresh local kernel in 3.5 minutes with TensorFlow 2.21.0. It verifies preprocessing, disjoint text groups, restored best validation-loss weights, metric calculations and prediction equality after model reload. Recorded execution versions: tensorflow==2.21.0, numpy==2.5.4, pandas==3.0.6, scikit-learn==1.9.1, matplotlib==3.11.2, ipykernel==7.4.0, keras==3.15.1, nbformat==5.11.1, nbclient==0.11.0. Timings and scores can vary by device/package version; Colab execution has not been tested.

## Run and repository contents

Open [news_classification.ipynb](news_classification.ipynb), place the trusted dataset in `data/`, install `requirements.txt` and run all cells. Colab also supports uploading the file directly to its working directory. Full VS Code and Colab instructions: [RUN_IT.md](RUN_IT.md).

- `news_classification.ipynb`: complete experiment with executed outputs.
- `requirements.txt`: supported package ranges for local execution.
- `DATASET.md`: dataset source, structure and local loading instructions.
- `.gitignore`: excludes generated reports, models, archives and local environments.

Execution generates `reports/` with metric tables and diagnostics, alongside the `.keras` model, preprocessing statistics/vocabulary, test predictions and `news_results.zip`. These are excluded from Git; download them locally to reuse the trained network. The BoW vocabulary mapping is not supplied, so raw-text-only deployment is outside the scope of this experiment.

## Limitations

The supplied BoW vocabulary was extracted from the full collection, including held-out entries; this experiment is not fully inductive preprocessing. A fresh evaluation would rebuild BoW from training texts. Identical-text grouping does not rule out near-duplicates or shared stories. A random split measures held-out entries from this collection, rather than future years or a different news source. Month is a standardized numeric input rather than a cyclic encoding.

There is one split and one seed, with repeated validation use for checkpoint, configuration and threshold selection. Validation scores can therefore be optimistic. The collection has previously been used for development; this test is an internal evaluation, not an untouched external benchmark. Dropout, L2 and early stopping do not establish that overfitting is absent. No input-branch ablation is performed, so the contribution of each representation is not quantified.
