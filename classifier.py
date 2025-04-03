from google import genai
from google.genai import types
from google.api_core import retry
from dotenv import load_dotenv
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import tqdm
from tqdm.rich import tqdm as tqdmr
import keras
from keras import layers
import warnings
import os
import email
import re
import numpy as np


load_dotenv()
is_retriable = lambda e: (isinstance(e, genai.errors.APIError) and e.code in {429, 503})
warnings.filterwarnings("ignore", category=tqdm.TqdmExperimentalWarning)
tqdmr.pandas()

def preprocess_newsgroup_row(data):
    msg = email.message_from_string(data)
    text = f"{msg['Subject']}\n\n{msg.get_payload()}"
    text = re.sub(r"[\w\.-]+@[\w\.-]+", "", text)
    text = text[:5000]
    return text

def preprocess_newsgroups_data(newsgroups_dataset):
    df = pd.DataFrame(
        {
            "Text": newsgroups_dataset.data,
            "Label": newsgroups_dataset.target,
        }
    )
    df["Text"] = df["Text"].apply(preprocess_newsgroup_row) 
    df["Class Name"] = df["Label"].apply(lambda x: newsgroups_dataset.target_names[x])
    return df

def sample_data(df, num_samples, classes_to_keep):
    df = (
        df.groupby("Label")[df.columns]
        .apply(lambda x: x.sample(num_samples))
        .reset_index(drop=True)  
    )
    df = df[df["Class Name"].str.contains(classes_to_keep)]
    df["Class Name"] = df["Class Name"].astype("category")
    df["Encoded Label"] = df["Class Name"].cat.codes
    return df

@retry.Retry(predicate=is_retriable, timeout=300.0)
def embed_fn(text: str) -> list[float]:
    response = client.models.embed_content(
        model="models/text-embedding-004",
        contents=text,
        config=types.EmbedContentConfig(
            task_type="classification",
        ),
    )
    return response.embeddings[0].values

def create_embeddings(df):
    df["Embeddings"] = df["Text"].progress_apply(embed_fn)
    return df

def build_classification_model(input_size, num_classes) -> keras.Model:
    return keras.Sequential(
        [
            layers.Input([input_size], name="embedding_inputs"),
            layers.Dense(input_size, activation="relu", name="hidden"),
            layers.Dense(num_classes, activation="softmax", name="output_probs"),
        ]
    )

def make_prediction(text):
    embedded = embed_fn(text)
    inp = np.array([embedded])
    [result] = classifier.predict(inp)
    return result



if __name__=="__main__":
    api_key = os.getenv('GOOGLE_API_KEY')
    client = genai.Client(api_key=api_key)

    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_test = fetch_20newsgroups(subset='test')

    df_train = preprocess_newsgroups_data(newsgroups_train)
    df_test = preprocess_newsgroups_data(newsgroups_test)

    TRAIN_NUM_SAMPLES = 100
    TEST_NUM_SAMPLES = 25
    CLASSES_TO_KEEP = "sci"

    df_train = sample_data(df_train, TRAIN_NUM_SAMPLES, CLASSES_TO_KEEP)
    df_test = sample_data(df_test, TEST_NUM_SAMPLES, CLASSES_TO_KEEP)

    df_train = create_embeddings(df_train)
    df_test = create_embeddings(df_test)

    embedding_size = len(df_train["Embeddings"].iloc[0])
    classifier = build_classification_model(embedding_size, len(df_train["Class Name"].unique()))
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    y_train = df_train["Encoded Label"]
    x_train = np.stack(df_train["Embeddings"])
    y_val = df_test["Encoded Label"]
    x_val = np.stack(df_test["Embeddings"])
    early_stopping = keras.callbacks.EarlyStopping( monitor="accuracy", patience=3)
    history = classifier.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
    )

    ############# Prediction #############
    new_text = """
First-timer looking to get out of here.

Hi, I'm writing about my interest in travelling to the outer limits!

What kind of craft can I buy? What is easiest to access from this 3rd rock?

Let me know how to do that please.
"""
    result = make_prediction(new_text)

    for idx, category in enumerate(df_test["Class Name"].cat.categories):
        print(f"{category}: {result[idx] * 100:0.2f}%")