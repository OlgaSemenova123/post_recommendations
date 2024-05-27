import os
import pickle
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
# from schema import PostGet
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.display.max_columns = 500

app = FastAPI()
class PostGet(BaseModel):
    id: int
    text: str
    topic: str

    class Config:
        orm_mode = True

# Загрузка модели
def get_model_path(path: str) -> str:
    if os.environ.get("IS_LMS") == "1":  # проверяем где выполняется код в лмс, или локально. Немного магии
        MODEL_PATH = '/workdir/user_input/model'
    else:
        MODEL_PATH = path
    return MODEL_PATH


def load_models():
    model_path = get_model_path("catboost_model")
    cat_features = ['topic', 'country', 'city', 'os', 'source']
    model = CatBoostClassifier(cat_features=cat_features)
    model.load_model(model_path)
    return model



# Загрузка признаков
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


def load_features() -> pd.DataFrame:
    query = "SELECT * FROM olg_semenova_lesson_22"
    return batch_load_sql(query)


# Сохранение признаков в базу данных
def save_features(features: pd.DataFrame):
    engine = create_engine(
        "postgresql://robot-startml-ro:pheiph0hahj1Vaif@"
        "postgres.lab.karpov.courses:6432/startml"
    )
    features.to_sql(
        name='olg_semenova_lesson_22',
        con=engine,
        if_exists='replace',
        index=False
    )


# Загрузка модели и признаков вне endpoint
model = load_models()
df_features = load_features()

df_post = batch_load_sql('SELECT * FROM public.post_text_df')
"""
df = df_post.copy()
df['text'] = df['text'].fillna('')
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_dense = tfidf_matrix.todense()
tfidf_df = pd.DataFrame(tfidf_dense, columns=feature_names)
# Вычисляем сумму значений по каждой колонке в матрице TF-IDF
column_sums = tfidf_df.sum()

# Определяем индексы колонок, где сумма значений не больше 50
columns_to_remove = column_sums[column_sums <= 70].index

# Удаляем колонки с такими индексами из DataFrame TF-IDF
tfidf_df = tfidf_df.drop(columns=columns_to_remove)
d = df_post[['post_id', 'topic']]
tfidf_data = pd.concat([d, tfidf_df], axis=1)
df_features = pd.merge(df_features, tfidf_data, left_on='post_id', right_on='post_id', how='left')"""

def generate_posts(top_posts, df_post, limit):
    for post_id in top_posts[:limit]:
        post_data = df_post[df_post['post_id'] == post_id]
        if not post_data.empty:
            text = post_data['text'].values[0]
            topic = post_data['topic'].values[0]
            yield PostGet(id=post_id, text=text, topic=topic)

@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 5) -> List[PostGet]:
    # Отбор признаков для конкретного user_id
    user_features = df_features[df_features['user_id'] == id]

    # Прогноз
    user_pred_proba = model.predict_proba(user_features)[:, 1]
    top_posts = user_features['post_id'].iloc[np.argsort(user_pred_proba)[::-1]].tolist()

    # Возвращаем ТОП-5 постов
    result = generate_posts(top_posts, df_post, limit)

    return list(result)

