import psycopg2
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

connection = psycopg2.connect(
    database="startml",
    user="robot-start****",
    password="******",
    host="postgres.lab.karpov*****",
    port=6432
)

cursor = connection.cursor()
cursor.execute("""SELECT * FROM public.user_data""")
result = cursor.fetchall()

df_user = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

cursor.execute("""SELECT * FROM public.post_text_df""")
result = cursor.fetchall()

df_post = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

cursor.execute("""SELECT * FROM public.feed_data limit 200000""")
result = cursor.fetchall()

df_feed = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])
cursor.close()
connection.close()
df_feed = df_feed.sort_values("timestamp")
from sklearn.feature_extraction.text import TfidfVectorizer
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

data = pd.merge(df_feed, df_user, left_on='user_id', right_on='user_id', how='left')
# Преобразуем столбец timestamp в формат datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Выделяем год в отдельный столбец
data['year'] = data['timestamp'].dt.year

# Выделяем номер месяца в отдельный столбец
data['month'] = data['timestamp'].dt.month

# Выводим результат
data = data.drop('timestamp', axis=1)
data.drop('action', axis=1)
data = pd.merge(data, tfidf_data, left_on='post_id', right_on='post_id', how='left')

from catboost import CatBoostClassifier
catboost_model = CatBoostClassifier(learning_rate=0.02)

X_train = data.iloc[:-80000].copy().drop(['target', 'action'], axis=1)
X_test = data.iloc[-80000:].copy().drop(['target', 'action'], axis=1)
y_train = data['target'].iloc[:-80000].copy()
y_test = data['target'].iloc[-80000:].copy()
cat_features = ['topic', 'age', 'country', 'city', 'os', 'source']
catboost_model.fit(X_train, y_train, cat_features)
from sklearn.metrics import accuracy_score
y_train_pred = catboost_model.predict(X_train)
y_test_pred = catboost_model.predict(X_test)
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy on train set: {train_accuracy}')
print(f'Accuracy on test set: {test_accuracy}')
y_test_array = y_test.to_numpy()
probs = catboost_model.predict_proba(X_test)
# Получаем индексы вероятностей, отсортированные по убыванию
sorted_indexes = np.argsort(probs, axis=1)[:, ::-1]

# Проверяем, содержат ли истинные метки в топ-5 предсказанных классов
hits = []
for i in range(len(y_test_array)):
    true_label = y_test_array[i]
    top5_labels = sorted_indexes[i, :5] + 1  # Добавляем 1 к индексам
    if true_label in top5_labels:
        hits.append(1)
    else:
        hits.append(0)

# Вычисляем HitRate@5
hit_rate_at_5 = np.mean(hits)
print("HitRate@5:", hit_rate_at_5)
catboost_model.save_model('catboost_model',
                           format="cbm")

from_file = CatBoostClassifier()  # здесь не указываем параметры, которые были при обучении, в дампе модели все есть

from_file.load_model("catboost_model")

from_file.predict(X_train)
