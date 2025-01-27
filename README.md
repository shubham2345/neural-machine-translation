

```markdown
# Neural Machine Translation

This repository contains scripts and resources for training Neural Machine Translation (NMT) models using Fairseq on the IWSLT’13 French-English dataset.

## 🚀 Environment Setup

To set up the environment, follow these steps:

```bash
conda create -n fairseq_203 python=3.9
conda activate fairseq_203
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install pip==23.0.1
pip install --editable ./
pip install fastBPE sacremoses subword_nmt
git clone git@github.com:moses-smt/mosesdecoder.git
git clone git@github.com:rsennrich/subword-nmt.git
```

---

## 📂 Data Preprocessing

We use `processing.py` for dataset preparation. The script includes a `--bpe` flag to apply Byte Pair Encoding (BPE):

- **Part A (With BPE)**:
  - Run:  
    ```bash
    python processing.py --bpe
    ```
  - Output: `outputs/token_counts.log`
  - Creates: `data-bin/fr-en/`

- **Part B (Without BPE)**:
  - Run:  
    ```bash
    python processing.py
    ```
  - Output: `outputs/token_counts.log`
  - Creates: `data-bin/fr-en-no-bpe/`

---

## 🏗️ Model Training

### **Part A (With BPE)**
| Model | Script | Output |
|--------|----------------------------|--------------------------------|
| Transformer | `transformer_train_parta.sh` | `outputs/transformer_output_parta.txt` |
| CNN | `cnn_train_parta.sh` | `outputs/cnn_output_parta.txt` |

### **Part B (Without BPE)**
| Model | Script | Output |
|--------|----------------------------|--------------------------------|
| Transformer | `transformer_train_partb.sh` | `outputs/transformer_output_partb.txt` |
| CNN | `cnn_train_partb.sh` | `outputs/cnn_output_partb.txt` |

---

## 🚧 Part C: Work In Progress
Scripts for **Part C** are still being developed and will be submitted before the final deadline. If you have any suggestions, please review the following scripts and provide feedback:

- **Transformer:** `transformer_train_partc.sh`
- **CNN:** `cnn_train_partc.sh`

---

## 📌 Notes
- `data-bin/fr-en/` is created with **BPE applied**.
- `data-bin/fr-en-no-bpe/` is created **without BPE**.
- **Fairseq is used for model training**.
- The dataset consists of TED talk transcripts and manual translations.

---

## 🔥 Results & Evaluation

To evaluate model performance, we use **SacreBLEU** for BLEU score computation:

```bash
fairseq-generate data-bin/fr-en \
    --path checkpoints/transformer_model_parta/checkpoint_best.pt \
    --batch-size 128 --beam 5 --remove-bpe \
    --scoring sacrebleu
```

- **Baseline Transformer BLEU Score:** `37.25`
- **Baseline CNN BLEU Score:** `30.67`
- **Part B (No BPE) Transformer BLEU Score:** `35.76`
- **Part B (No BPE) CNN BLEU Score:** `30.68`

---

## 📌 Future Work

- Experiment with different Transformer architectures.
- Tune hyperparameters (dropout, learning rate, layers).
- Optimize BPE settings for better translation quality.


