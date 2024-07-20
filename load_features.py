import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sqlalchemy import create_engine

def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    engine = create_engine(
        "postgresql://robot-start******"
        "postgres.lab.karpov**********"
    )
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)
df_feed_all_user = batch_load_sql('''WITH ranked_posts AS (
    SELECT user_id, post_id, timestamp, target,
           ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY target DESC, post_id) AS post_rank
    FROM feed_data)
SELECT user_id, post_id, timestamp, target
FROM ranked_posts
WHERE post_rank <= 5''')
df_user = batch_load_sql('SELECT * FROM public.user_data')

data = pd.merge(df_feed_all_user, df_user, left_on='user_id', right_on='user_id', how='left')
# Преобразуем столбец timestamp в формат datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Выделяем год в отдельный столбец
data['year'] = data['timestamp'].dt.year

# Выделяем номер месяца в отдельный столбец
data['month'] = data['timestamp'].dt.month

# Выводим результат
data = data.drop('timestamp', axis=1)
data = data.drop('target', axis=1)

df_post = batch_load_sql('SELECT * FROM public.post_text_df')
df = df_post.copy()
df['text'] = df['text'].fillna('')
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_dense = tfidf_matrix.todense()
tfidf_df = pd.DataFrame(tfidf_dense, columns=feature_names)

# Применяем PCA для уменьшения размерности до 10
pca = PCA(n_components=10)
pca_matrix = pca.fit_transform(tfidf_matrix.toarray())

# Преобразуем pca_matrix в датафрейм
df_pca = pd.DataFrame(data=pca_matrix, columns=[f'PC{i+1}' for i in range(pca_matrix.shape[1])])
d = df_post[['post_id', 'topic']]
tfidf_data = pd.concat([d, df_pca], axis=1)
data = pd.merge(data, tfidf_data, left_on='post_id', right_on='post_id', how='left')
print(data)
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostClassifier

# Создание соединения с базой данных
engine = create_engine(
    "postgresql://robot-start*******"
    "postgres.lab.karpov*******"
)

# Сохранение признаков в таблицу
def save_features(features: pd.DataFrame):
    features.to_sql(
        name='******lesson_22',
        con=engine,
        if_exists='replace',
        index=False
    )

# Загрузка признаков из таблицы
def load_features() -> pd.DataFrame:
    query = "SELECT * FROM *******lesson_22"
    return batch_load_sql(query)


save_features(data)
