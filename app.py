from flask import Flask, request, render_template
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN = 256
device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
model.eval()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pred = torch.argmax(probs).item()

    return {
        "label": "REAL" if pred == 1 else "FAKE",
        "confidence": float(probs[pred]) * 100,
        "fake_prob": float(probs[0]) * 100,
        "real_prob": float(probs[1]) * 100
    }

@app.route("/")
def home():
    return render_template("index.html", result=None)

@app.route("/predict", methods=["POST"])
def predict_route():
    text = request.form["news"]
    result = predict(text)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run()