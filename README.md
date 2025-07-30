## 🌐 Language Translation Using Deep Learning

An end-to-end deep learning project for translating text between languages using neural machine translation (NMT) models.

---

## ⚙️ Features

- Sequence-to-sequence (seq2seq) architecture with LSTM or GRU
- Attention mechanism support (Bahdanau or Luong)
- Data preprocessing: tokenization, padding, vocabulary creation
- BLEU score evaluation
- Custom sentence inference
- Jupyter Notebook for interactive experimentation

---

## 📦 Requirements

```bash
python >= 3.7
numpy
pandas
tqdm
nltk
tensorflow or torch
sacrebleu (for BLEU scoring)
````

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🗂️ Project Structure

```
language-translation-using-deep-learning/
│
├── data/                   # Contains training/validation data
├── src/                    # Core Python scripts
│   ├── preprocess.py       # Data preprocessing utilities
│   ├── model.py            # Encoder-decoder models
│   ├── train.py            # Model training logic
│   ├── evaluate.py         # BLEU scoring and metrics
│   └── infer.py            # Translation inference
├── notebooks/              # Jupyter Notebooks for demo/experiments
├── checkpoints/            # Trained model weights (optional)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Preprocess your data

```bash
python src/preprocess.py \
  --input-src data/train.en \
  --input-tgt data/train.fr \
  --output-dir data/processed
```

### 2. Train the model

```bash
python src/train.py \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 64 \
  --save-dir checkpoints/
```

### 3. Evaluate BLEU Score

```bash
python src/evaluate.py \
  --model checkpoints/model_final.pth \
  --data-dir data/processed \
  --metric bleu
```

### 4. Translate Custom Sentences

```bash
python src/infer.py \
  --model checkpoints/model_final.pth \
  --sentence "How are you today?"
```

---

## 🧠 Supported Models

* ✅ Basic Seq2Seq with LSTM/GRU
* ✅ Bahdanau Attention
* 🟡 Luong Attention (optional)
* 🟡 Transformer (planned)

---

## 📊 Evaluation

| Model        | BLEU Score |
| ------------ | ---------- |
| LSTM Seq2Seq | 24.5       |
| + Attention  | 28.7       |
| Transformer  | TBD        |

Visualize output translations and metrics inside the Jupyter notebooks.

---

## 📈 Demo Notebook

* Run `notebooks/demo.ipynb` to try out the model interactively.
* Input custom sentences, view predictions, attention heatmaps (if applicable).

---

## 🛣️ Future Improvements

* [ ] Transformer implementation
* [ ] Web app demo (Flask or Streamlit)
* [ ] Pre-trained multilingual embeddings
* [ ] Beam search decoding

---

## 🤝 Contributing

1. Fork this repo
2. Create your branch: `git checkout -b my-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin my-feature`
5. Open a Pull Request

---

## 📜 License

Licensed under the MIT License. See `LICENSE` for more info.

---

## 🙋‍♀️ Author

Created with 💻 by [Jheelam Hossain](https://github.com/jheelamH)
Feel free to connect or contribute!

