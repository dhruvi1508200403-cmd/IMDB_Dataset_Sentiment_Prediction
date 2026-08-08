# 🎬 CineSentiment — IMDB Review Sentiment Tug of War

A Streamlit app that predicts whether a movie review is **Positive** or **Negative** using a Simple RNN trained on the IMDB dataset — visualized as a live tug-of-war between Team Positive and Team Negative, with the entire site's mood color shifting as more reviews are analyzed.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/TensorFlow-Keras-orange?logo=tensorflow" />
  <img src="https://img.shields.io/badge/Model-Simple%20RNN-green" />
</p>

---

## ✨ Features

- **Real-time sentiment prediction** on any movie review, powered by a Simple RNN trained on the IMDB dataset
- **Tug-of-war visualization** — Team Negative and Team Positive pull a rope, with the knot's position and pull strength driven directly by the model's prediction score
- **Site-wide mood color** — every review analyzed nudges a running average, which tints the entire app's background between red and green
- **Battle log** — a running history of every review submitted in the session, with sentiment and score
- **Live stats sidebar** — positive/negative counts and an overall mood meter
- **Quick example reviews** — one-click rave, brutal, and mixed reviews to try the model instantly
- **Confetti & snow effects** for strongly positive / strongly negative predictions

---

## 🖥️ Demo

Run locally and open the app in your browser — type a review, hit **"⚔️ Send it into battle,"** and watch the prediction score, sentiment badge, and tug-of-war animation update instantly.

---

## 🧠 How It Works

1. **Preprocessing** — the review is lowercased, stripped of punctuation, and tokenized
2. **Encoding** — each word is mapped to its integer index from the IMDB word index (offset by 3, per Keras convention); unknown or high-index words are mapped to `2` (the `<UNK>` token)
3. **Padding** — the encoded sequence is padded/truncated to a fixed length of `500`
4. **Inference** — the padded sequence is passed through the trained Simple RNN, which outputs a score between `0` (very negative) and `1` (very positive)
5. **Classification** — a score `>= 0.5` is labeled **Positive**, otherwise **Negative**

```python
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input, verbose=0)
    score = float(prediction[0][0])
    sentiment = "Positive" if score >= 0.5 else "Negative"
    return sentiment, score
```

---

## 📂 Project Structure

```
imdb_dataset_sentiment_prediction/
├── app.py                     # Streamlit application (UI + inference)
├── simple_rnn_imdb_model.h5   # Trained Simple RNN model
├── requirements.txt           # Python dependencies
└── README.md                  # You are here
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/dhruvi1508200403-cmd/imdb_dataset_sentiment_prediction.git
cd imdb_dataset_sentiment_prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Make sure the model file is present
Place `simple_rnn_imdb_model.h5` in the project root (same folder as `app.py`).

### 5. Run the app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📦 requirements.txt

```
streamlit
tensorflow
numpy
```

> **Note:** Do not add `re` to `requirements.txt` — it's a built-in Python standard-library module, not a PyPI package, and including it will break dependency resolution on deployment platforms like Streamlit Community Cloud.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your GitHub account
3. Select this repository, set the branch to `main` and the main module to `app.py`
4. Deploy — Streamlit Cloud will install dependencies from `requirements.txt` automatically

**Important:** `simple_rnn_imdb_model.h5` must be committed to the repository (or fetched at runtime from cloud storage) since Streamlit Cloud only has access to files in the repo.

---

## 🏗️ Model Details

| Property | Value |
|---|---|
| Architecture | Simple RNN |
| Training Dataset | IMDB Movie Reviews (Keras built-in) |
| Vocabulary Size | 10,000 most frequent words |
| Max Sequence Length | 500 tokens |
| Output | Sigmoid score in `[0, 1]` |
| Classification Threshold | `>= 0.5` → Positive |

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** — web app framework
- **[TensorFlow / Keras](https://www.tensorflow.org/)** — model training & inference
- **NumPy** — numerical operations
- **HTML/CSS (embedded via `streamlit.components.v1`)** — the animated tug-of-war visualization

---

## 🚧 Known Limitations

- The model only recognizes the 10,000 most frequent words from the IMDB vocabulary; rarer words are mapped to an `<UNK>` token, which can affect predictions on niche or technical reviews
- Sarcasm, negation edge cases, and mixed-sentiment reviews may be misclassified, as is common with simple RNN architectures
- `imdb.get_word_index()` downloads a small JSON file on first run — an internet connection is required the first time the app starts

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Feel free to open a pull request or file an issue.

---

## 📄 License

This project is open source and available under the [GNU](LICENSE).
