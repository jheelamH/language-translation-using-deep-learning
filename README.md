
# 🌐 Language Translation Using Deep Learning  

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)  
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)  
![License](https://img.shields.io/badge/License-MIT-pink.svg)  
![Made with Love](https://img.shields.io/badge/Made%20with-💻%20&%20coffee-ff69b4)  

An end-to-end **neural machine translation (NMT)** project for translating text between languages using **seq2seq models with attention**.  

Bring text from one language 📝 to another 🌏 using deep learning magic ✨.  

---

## ⚙️ Features  

- Sequence-to-sequence (Seq2Seq) with **LSTM / GRU**  
- **Attention mechanism** support (Bahdanau & optional Luong)  
- Data preprocessing: tokenization, padding, vocab creation  
- BLEU score evaluation 📊  
- Translate **custom sentences** interactively  
- Jupyter Notebook for experimentation 🖥️  

---

## 📦 Requirements  

```bash
python >= 3.7
numpy
pandas
tqdm
nltk
tensorflow or torch
sacrebleu
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
├── data/                   # Training/validation data
├── src/                    # Core Python scripts
│   ├── preprocess.py       # Data preprocessing
│   ├── model.py            # Encoder-decoder models
│   ├── train.py            # Training logic
│   ├── evaluate.py         # BLEU scoring & metrics
│   └── infer.py            # Translation inference
├── notebooks/              # Jupyter notebooks for demos
├── checkpoints/            # Saved model weights
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1️⃣ Preprocess your data

```bash
python src/preprocess.py \
  --input-src data/train.en \
  --input-tgt data/train.fr \
  --output-dir data/processed
```

### 2️⃣ Train the model

```bash
python src/train.py \
  --data-dir data/processed \
  --epochs 20 \
  --batch-size 64 \
  --save-dir checkpoints/
```

### 3️⃣ Evaluate BLEU Score

```bash
python src/evaluate.py \
  --model checkpoints/model_final.pth \
  --data-dir data/processed \
  --metric bleu
```

### 4️⃣ Translate Custom Sentences

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

Visualize translations and attention heatmaps inside notebooks.

---

## 📈 Demo Notebook

* Run `notebooks/demo.ipynb` for interactive usage
* Input custom sentences, view predictions & attention heatmaps

---

## 🛣️ Future Improvements

* [ ] Transformer implementation
* [ ] Web app demo (Flask / Streamlit)
* [ ] Pre-trained multilingual embeddings
* [ ] Beam search decoding for better translations

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push to branch: `git push origin my-feature`
5. Open a Pull Request

---

## 📜 License

MIT — free to use, share, and remix 🌸

---

## 🙋‍♀️ About

Girl-coded 💻 with love for AI & deep learning.
Explore, experiment, and contribute! 🌟


