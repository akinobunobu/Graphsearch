import json
import spacy
import numpy as np

# SpaCyの日本語モデルをロード
nlp = spacy.load("ja_core_news_md")

def compute_average_embedding(text):
    """文章内の全単語の平均埋め込みを計算"""
    doc = nlp(text)
    vectors = [token.vector for token in doc if token.has_vector]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(nlp("dummy").vector.shape)

def vector_search(query, graph_data_file, top_n=3):
    """ベクトル検索"""
    with open(graph_data_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    product_embeddings = {
        node["node"]: np.array(node["embedding"]) for node in graph_data["nodes"] if "embedding" in node
    }

    query_vector = compute_average_embedding(query)

    product_scores = {}
    for product_name, product_vector in product_embeddings.items():
        norm_query = np.linalg.norm(query_vector)
        norm_product = np.linalg.norm(product_vector)
        if norm_query > 0 and norm_product > 0:
            similarity = np.dot(query_vector, product_vector) / (norm_query * norm_product)
            product_scores[product_name] = similarity

    top_results = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_results

if __name__ == "__main__":
    # 入力データファイル
    graph_data_file = "output/embeddings_direct2.2_entity2.0_similarity0.5.json"

    # ユーザー入力クエリ
    query = input("検索クエリを入力してください: ")

    # 上位N件（固定で3件）
    top_n = 3

    # ベクトル検索の結果
    print("\nベクトル検索結果（上位3件）:")
    vector_results = vector_search(query, graph_data_file, top_n)
    for rank, (product, score) in enumerate(vector_results, start=1):
        print(f"{rank}. {product} (スコア: {score:.4f})")
