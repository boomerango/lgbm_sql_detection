import sys
import pickle
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer


# load fuzzed to compare lgbm_model_fuzzed.txt and tfidf_vectorizer_fuzzed.pkl

bst = lgb.Booster(model_file='lgbm_model.txt')

# Load the vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def run_inference(texts):
    # Vectorize the input text
    X_tfidf = vectorizer.transform(texts)
    y_pred_prob = bst.predict(X_tfidf)
    
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]
    return y_pred

if not sys.stdin.isatty():
    input_text = sys.stdin.read().strip()    
    # Assuming each line is a separate log entry
    # texts = input_text.splitlines()
    texts = [input_text.strip()]
    predictions = run_inference(texts)
    
    mapper = {0: "Normal", 1: "Anomaly"}
    for i, pred in enumerate(predictions):
        print(f"Text: {texts[i]}")
        print(f"Prediction: {mapper[pred]}")
        print("-" * 50)
else:
    print("No input received from stdin. Usage: cat logs.txt | python lgbm_inference.py")
