import re

import nltk
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from keras.src.utils import to_categorical
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
from gensim.downloader import load as gensim_load
from sklearn.linear_model import LogisticRegression
from tcn import TCN

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
from collections import Counter

from transformers import TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer, Trainer
from xgboost import XGBClassifier


def init_nltk():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    nltk.download('wordnet')

def sampler(df, sample_size=-1, group_by_col = "source", stratified = True):
    """
    :param df:
    :param sample_size: sample size required in each class
    ex: n = 1000, it means 1000 records in each class
    :param group_by_col:
    :param stratified:
    :return:
    """
    if sample_size< -1:
        return df
    else:
        if stratified:
            return df.groupby(group_by_col,group_keys=False).apply(lambda x: x.sample(sample_size)) # here x is basically each group
        else:
            return df.sample(sample_size)


def preprocess_text(df):
    df["text"] = df["text"].apply(_preprocessor)
    return df

def _preprocessor(text):
    if pd.isnull(text):
        return ""
    # Lowercase
    text = text.lower()

    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub('\s+', ' ', text)
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def gru_runner(df, out_of_fold):

    x, em_mat, vocab = create_glove_embeddings_from_texts(df["text"])

    xt, xte, yt, yte = train_test_split(
        x, df["source"], test_size=0.2, random_state=42, stratify=df["source"]
    )

    x_preds, x_probs, t_x_probs, x_report = train_xgboost(xt, xte, yt, yte)
    print("XGBoost results")
    print(x_report)

    gru_probs, t_gru_probs,gru_report = train_gru_model(xt, xte, yt, yte, em_mat, vocab)
    print("GRU results")
    print(gru_report)
    tcn_probs, t_tcn_probs, tcn_report = train_tcn_model(xt, xte, yt, yte, em_mat, vocab)
    print("TCN results")
    print(tcn_report)

    stackgru = np.column_stack([t_gru_probs, t_x_probs])

    stack_gru_test = np.column_stack([gru_probs, x_probs])

    meta = LogisticRegression()
    meta.fit(stackgru, yt)

    x_tcn_preds = meta.predict(stack_gru_test)
    x_gru_report = classification_report(yte, x_tcn_preds)

    stacktcn = np.column_stack([t_tcn_probs, t_x_probs])
    meta.fit(stacktcn, yt)

    stack_gru_test = np.column_stack([tcn_probs, x_probs])
    x_tcn_preds = meta.predict(stack_gru_test)
    x_tcn_report = classification_report(yte, x_tcn_preds)

    print("XGBoost_GRU results")
    print(x_gru_report)

    print("XGBoost_TCN results")
    print(x_tcn_report)
    return

def tcn_runner(df):

    x, em_mat, vocab = create_glove_embeddings_from_texts(df["text"])



    return



def create_glove_embeddings_from_texts(texts, max_num_words=20000, embedding_dim=300, max_seq_len=100):

    # Split strings into tokens
    tokenized_texts = [t.split() for t in texts]

    # Build vocabulary
    all_tokens = [token for doc in tokenized_texts for token in doc]
    most_common = Counter(all_tokens).most_common(max_num_words)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}  # 0 reserved for padding

    print(f"Vocab size: {len(vocab)}")

    # Convert to sequences
    sequences = [[vocab[token] for token in doc if token in vocab] for doc in tokenized_texts]
    x = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')

    # Load GloVe 300D from gensim
    print("Loading GloVe 300D embeddings via gensim...")
    glove_vectors = gensim_load("glove-wiki-gigaword-300")
    print("GloVe loaded successfully.")

    # Build embedding matrix
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for word, i in vocab.items():
        if word in glove_vectors:
            embedding_matrix[i] = glove_vectors[word]

    return x, embedding_matrix, vocab


def train_xgboost(xt, xte, yt, yte,use_gpu=True):

    # Initialize XGBoost classifier
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        tree_method="auto",  # GPU/CPU selection
        use_label_encoder=False
    )

    # Train
    model.fit(xt, yt)

    print(model.classes_)

    # Predict
    preds = model.predict(xte)
    probs = model.predict_proba(xte)

    # Classification report
    report = classification_report(yte, preds)

    train_probs = model.predict_proba(xt)

    return preds, probs[:,1], train_probs[:,1],report

def train_gru_model(xt, xte, yt, yte, embedding_matrix, vocab, max_seq_len=100, embedding_dim=300, epochs=10, batch_size=64):


    model = Sequential([
        Embedding(
            input_dim=len(vocab) + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=False
        ),
        GRU(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(
        xt, yt,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    probs = model.predict(xte)
    y_pred = (probs > 0.5).astype("int32")

    report_dict = classification_report(yte, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    train_probs = model.predict(xt)
    return probs, train_probs, report_df

def train_tcn_model(xt, xte, yt, yte, embedding_matrix, vocab, max_seq_len=100, embedding_dim=300, epochs=10, batch_size=64):

    model = Sequential([
        Embedding(
            input_dim=len(vocab) + 1,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_seq_len,
            trainable=False
        ),
        TCN(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    model.fit(
        xt, yt,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    probs = model.predict(xte)
    y_pred = (probs > 0.5).astype("int32")

    report_dict = classification_report(yte, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    train_probs = model.predict(xt)
    return probs, train_probs,report_df

def transformer_runner(x, y):
    xt, xte, yt, yte = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # model_name = "distilbert-base-uncased"
    model_names = ["FacebookAI/roberta-base", "distilbert-base-uncased"]


    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        train_dataset = tokenize_dataset(tokenizer, xt, yt)
        test_dataset = tokenize_dataset(tokenizer, xte, yte)
        trainer, model, _ = build_trainer(model_name, train_dataset, test_dataset)

        trainer.train()

        results = trainer.evaluate()
        print(f"Results for {model}:")
        print(results)

        get_classification_report(trainer, test_dataset, model_name, xte)




    return


def build_trainer(model_name, train_dataset, eval_dataset, output_dir="./results",
                  learning_rate=2e-5, batch_size=8, epochs=3):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # NVIDIA GPU
        print("Using CUDA (NVIDIA GPU)")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        DEVICE = torch.device("mps")  # Apple M1/M2 GPU
        print("Using MPS (Apple GPU)")
    else:
        DEVICE = torch.device("cpu")
        print("Using CPU")
    model.to(DEVICE)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer, tokenizer, model

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def get_classification_report(trainer, eval_dataset, name, xte):
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids
    print(f"\n===== CLASSIFICATION REPORT {name} =====")
    print(classification_report(labels, preds))

    import json
    import os

    os.makedirs("res/utils3", exist_ok=True)

    results = []
    for text, true_label, pred_label in zip(xte, labels, preds):
        results.append({
            "text": text,
            "actual_label": true_label,
            "predicted_label": pred_label,
            "ismisclassified": bool(true_label != pred_label)
        })

    safe_model_name = name.replace("/", "_")
    with open(f"res/utils3/{safe_model_name}_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False, default=default_converter)

def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def tokenize_dataset(tokenizer, texts, labels, max_length=256):
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels
    })