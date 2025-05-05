# This project has 2 modules: a DeepFake detection module and a Fake News detection module

## Fake News Detection using BERT + LLM (2-Tier Pipeline)

This project builds a **2-tier fake news detection system** using the **LIAR dataset** and a fine-tuned **BERT classifier**, with optional LLM-based verification for low-confidence outputs.


### Dataset: LIAR
* **Source:**
  *William Wang, University of California, Santa Barbara*
  Dataset paper: *“Liar, Liar Pants on Fire: A New Benchmark Dataset for Fake News Detection”*
  [\[Paper Link\]](https://arxiv.org/abs/1705.00648)
  [\[Dataset Homepage\]](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)

* **Format:** Tab-separated values (.tsv) with **no header row**

* **Files:**

  * `train.tsv`
  * `valid.tsv`
  * `test.tsv`

* **Columns:**

| Index | Column Name | Description                                      |
| ----- | ----------- | ------------------------------------------------ |
| 0     | `id`        | Statement ID                                     |
| 1     | `label`     | One of 6 classes: true, false, etc.              |
| 2     | `statement` | The actual text of the claim                     |
| 3–13  | Metadata    | Speaker, party, context, prior truth stats, etc. |

---

### What the Notebook Does

The included Jupyter notebook:

1. **Loads and processes the LIAR dataset** from TSV files

2. **Maps 6-class labels to binary classes**:

   * `true`, `mostly-true`, `half-true` → `real` (1)
   * `false`, `barely-true`, `pants-fire` → `fake` (0)

3. **Fine-tunes a `bert-base-uncased` model** using HuggingFace Transformers

4. **Evaluates the model** on validation and test sets using accuracy, F1, precision, and recall

5. **(Optional)**: Outputs low-confidence predictions for secondary verification via an LLM (GPT + web search)

---

Here's a clean, no-code documentation snippet for your GitHub `README.md` that focuses purely on the data, process, and what the `DeepFake.ipynb` notebook does:

---

## Deepfake Detection Module Overview

The deepfake detection module classifies face images as either real or fake. It focuses on frame-level detection using cropped facial images extracted from video datasets. The goal is to build a lightweight yet effective binary classifier capable of identifying deepfakes based on facial artifacts.

### Dataset

I use a curated subset of the **Celeb-DF v2** dataset, which includes:

* **Celeb-real**: Authentic face videos of celebrities
* **Celeb-synthesis**: Deepfake versions of the real videos
* **Youtube-real**: Additional real face videos from YouTube

From these sources, a total of \~130,000 facial frames were extracted using MTCNN. Each video contributed up to 20 face crops, and these were labeled and organized into two classes: `real` and `fake`.

The face crops were then split into three sets:

* **Training**: 5,000 samples
* **Validation**: 1,000 samples
* **Test**: 1,000 samples

### Notebook Functionality (`DeepFake.ipynb`)

The notebook performs the following steps:

1. **Video Preprocessing**: Loads videos, samples frames, and extracts faces using MTCNN.
2. **Face Crop Storage**: Saves cropped face images into labeled folders (`real/` and `fake/`).
3. **Dataset Splitting**: Organizes the images into training, validation, and test sets.
4. **Model Setup and Training**: Initializes a pretrained ResNet18, modifies it for binary classification, and trains it using PyTorch.
5. **Evaluation**: Reports accuracy on validation and test sets.
6. **Model Saving**: Saves the best-performing model to disk.

This module achieves perfect accuracy on the selected subset, demonstrating the model's ability to learn from distinguishable deepfake artifacts in facial images.
