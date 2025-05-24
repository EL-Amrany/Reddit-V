import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess

df = pd.read_csv('test.csv')  # expects columns 'Post Title', 'viral', and the categorical features

post_titles = df['Post Title'].tolist()
actual_labels_all = df['viral'].tolist()

categorical_features = [
    'Crossposts Count',
    'Comment Sentiment',
    'category_encoded',
    'title_sentiment',
    'First Hour Upvotes',
    'Subreddit Subscribers',
    'day',
    'month',
    'year',
]

cat_features_all = df[categorical_features].to_dict(orient='records')

models = [
    "gemma3:1b", "gemma3", "gemma3:12b", "llama3.2", "llama3.2:1b", "llama3.1",
    "phi4-mini", "mistral", "moondream", "neural-chat", "starling-lm",
    "codellama", "llama2-uncensored", "llava", "granite3.2"
]

metrics_summary = []

for model in models:
    model_safe = model.replace(":", "_")  # Safe filename (colon → underscore)
    predictions = []
    total_posts = len(post_titles)

    print(f"\nStarting predictions with model {model}...")

    for idx, (title, features) in enumerate(zip(post_titles, cat_features_all), 1):
        print(f"Model {model} - Processing line {idx}/{total_posts}")

        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])

        prompt = (
            "Based on the following social media post and its related metadata, "
            "predict whether it will go viral or not.\n\n"
            f"Post Title: \"{title}\"\n"
            f"{feature_text}\n\n"
            "Answer only with 'Viral' or 'Not Viral'."
        )

        prediction = None
        for attempt in range(3):
            try:
                result = subprocess.run(
                    ["ollama", "run", model],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=15  # seconds
                )
                if result.returncode == 0:
                    response = result.stdout.strip()
                    if response:
                        prediction = response
                        break  # Exit retry loop on success
            except subprocess.TimeoutExpired:
                print(f"Timeout for model {model} on attempt {attempt + 1}, retrying...")
                continue
            except Exception as e:
                print(f"Error calling model {model} (attempt {attempt + 1}): {e}")
                continue

        if prediction:
            response_clean = prediction.strip().strip('.').capitalize()
            if response_clean.lower().startswith("viral"):
                prediction_label = "Viral"
            else:
                prediction_label = "Not Viral"
        else:
            prediction_label = None  # could not get prediction

        predictions.append(prediction_label)

    output_df = pd.DataFrame({
        "Post Title": post_titles,
        "Actual": actual_labels_all,
        "Predicted": predictions
    })
    output_df.to_csv(f"predictions_{model_safe}.csv", index=False)

    valid_indices = [i for i, pred in enumerate(predictions) if pred is not None]
    if not valid_indices:
        print(f"No valid predictions for model {model}, skipping metrics calculation.")
        continue

    y_true = [actual_labels_all[i] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]
    y_true_bin = [1 if label == "Viral" else 0 for label in y_true]
    y_pred_bin = [1 if label == "Viral" else 0 for label in y_pred]

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    metrics_summary.append({
        "Model": model,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Posts Processed": len(valid_indices)
    })

    print(f"Model {model} completed: {len(valid_indices)} posts processed.")

metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv("metrics_summary.csv", index=False)

print("\n All models processed. Metrics saved to metrics_summary.csv.")
