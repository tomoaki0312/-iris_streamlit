import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import numpy as np

# ランダムな文章を生成する関数
def generate_random_text():
    trouble_keywords = ["電話", "画面", "通話", "音声", "操作", "バッテリー", "充電", "表示", "メッセージ", "スピーカー"]
    trouble_actions = ["鳴らない", "映らない", "できない", "小さい", "効かない", "切れる", "点滅する", "反応しない", "聞こえない", "壊れた", "消えた", "大きすぎる"]
    
    trouble_keyword = random.choice(trouble_keywords)
    trouble_action = random.choice(trouble_actions)
    
    return f"{trouble_keyword}が{trouble_action}"

# サンプルデータを生成
sample_data = [(generate_random_text(), random.choice(["通話トラブル", "画面トラブル", "音声トラブル", "操作トラブル", "バッテリートラブル", "充電トラブル", "表示トラブル", "メッセージトラブル"])) for _ in range(10000)]

# データとラベルを分割
texts, labels = zip(*sample_data)

# Word2Vecモデルの作成
word2vec_model = Word2Vec([text.split() for text in texts], vector_size=100, window=5, min_count=1, workers=4)

# 文章をWord2Vecベクトルに変換
X_word2vec = [np.mean([word2vec_model.wv[word] for word in text.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0) for text in texts]

# データをトレーニングデータとテストデータに分割
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(X_word2vec, labels, test_size=0.2, random_state=42)

# ロジスティック回帰分類器の作成
classifier_word2vec = LogisticRegression(max_iter=1000)

# モデルのトレーニング
classifier_word2vec.fit(X_train_word2vec, y_train_word2vec)

# テストデータで評価
y_pred_word2vec = classifier_word2vec.predict(X_test_word2vec)

# 精度評価
print("Accuracy (Word2Vec):", accuracy_score(y_test_word2vec, y_pred_word2vec))
print("Classification Report (Word2Vec):\n", classification_report(y_test_word2vec, y_pred_word2vec))

# 新しい文章に対してカテゴリを予測
new_text_word2vec = ["画面が点滅してしまいます"]
new_text_word2vec_vectorized = np.mean([word2vec_model.wv[word] for word in new_text_word2vec[0].split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0).reshape(1, -1)
predicted_category_word2vec = classifier_word2vec.predict(new_text_word2vec_vectorized)

print("Predicted Category (Word2Vec):", predicted_category_word2vec[0])
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse
app = FastAPI()
# テンプレートディレクトリの指定
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 新しい文章に対してカテゴリを予測するエンドポイント
@app.post("/predict_category", response_class=HTMLResponse)
async def predict_category(request: Request, new_text: str = Form(...)):
    # 文章をWord2Vecベクトルに変換
    new_text_vectorized = np.mean([word2vec_model.wv[word] for word in new_text.split() if word in word2vec_model.wv] or [np.zeros(100)], axis=0).reshape(1, -1)
    
    # カテゴリを予測
    predicted_category = classifier_word2vec.predict(new_text_vectorized)[0]
    
    return templates.TemplateResponse("index.html", {"request": request, "predicted_category": predicted_category})