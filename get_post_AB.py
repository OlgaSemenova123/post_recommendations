import os
import pickle
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import hashlib
from loguru import logger
pd.options.display.max_columns = 500

app = FastAPI()


class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True


class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


def get_model_path(path: str, name: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        MODEL_PATH = f"/workdir/user_input/{name}"
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models(name: str):
    model_path = get_model_path(f"{name}", name)
    cat_features = ['topic', 'country', 'city', 'os', 'source']
    model = CatBoostClassifier(cat_features=cat_features)
    model.load_model(model_path)
    return model


def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 50000
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def load_features_control() -> pd.DataFrame:
    query = "SELECT * FROM olg_semenova_lesson_22"
    return batch_load_sql(query)


def load_features_test() -> pd.DataFrame:
    query = "SELECT * FROM olg_semenova_lesson_10"
    return batch_load_sql(query)


def generate_posts(top_posts, df_post, limit):
    for post_id in top_posts[:limit]:
        post_data = df_post[df_post['post_id'] == post_id]
        if not post_data.empty:
            text = post_data['text'].values[0]
            topic = post_data['topic'].values[0]
            yield PostGet(id=post_id, text=text, topic=topic)


def get_exp_group(user_id: int) -> str:
    temp_exp_group = int(int(hashlib.md5((str(user_id) + 'my_salt').encode()).hexdigest(), 16) % 100)
    if temp_exp_group <= 50:
        exp_group = "control"
    elif temp_exp_group > 50:
        exp_group = "test"
    return exp_group


def get_recommendations_control(user_features):
    # Implement the control model's recommendation logic here
    logger.info("predicting_control")
    user_pred_proba = model_control.predict_proba(user_features)[:, 1]
    top_posts = user_features['post_id'].iloc[np.argsort(user_pred_proba)[::-1]].tolist()
    return top_posts


def get_recommendations_test(user_features):
    logger.info("predicting_test")
    # Implement the test model's recommendation logic here (from the third block)
    user_pred_proba = model_test.predict_proba(user_features)[:, 1]
    top_posts = user_features['post_id'].iloc[np.argsort(user_pred_proba)[::-1]].tolist()
    return top_posts


model_control = load_models("catboost_model")
model_test = load_models("catboost_model (7)")
df_post = batch_load_sql('SELECT * FROM public.post_text_df')
df_features_control = load_features_control()
df_features_test = load_features_test()
logger.info("service is up and running")

@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> Response:

    exp_group = get_exp_group(id)

    if exp_group == "control":
        user_features = df_features_control[df_features_control['user_id'] == id]
        top_posts = get_recommendations_control(user_features)
    elif exp_group == "test":
        user_features = df_features_test[df_features_test['user_id'] == id]
        top_posts = get_recommendations_test(user_features)
    else:
        raise HTTPException(status_code=500, detail="Unexpected experiment group")

    result = generate_posts(top_posts, df_post, limit)
    return Response(exp_group=exp_group, recommendations=list(result))

