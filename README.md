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
