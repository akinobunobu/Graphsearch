import spacy
import json
import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# SpaCyの日本語モデルのロード
nlp = spacy.load("ja_core_news_md")

def load_dataset(dataset_file):
    """商品データセット（JSON形式）をロード"""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    return dataset

def compute_similarity(vec1, vec2):
    """コサイン類似度を計算"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def extract_relevant_nodes_with_similarity(input_text, dataset_nodes, node_embeddings, threshold=0.5):
    """
    入力テキスト内の動詞、名詞、形容詞を抽出し、
    近似度に基づいて商品データセット内のノードに関連付ける
    """
    doc = nlp(input_text)
    relevant_nodes = set()
    unmatched_nodes = []  # 一致しなかった単語のリスト

    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            lemma = token.lemma_
            if lemma in dataset_nodes:
                relevant_nodes.add(lemma)
            else:
                lemma_vector = nlp(lemma).vector
                unmatched_nodes.append((lemma, lemma_vector))

    return list(relevant_nodes), unmatched_nodes

def calculate_total_weights_with_similarity(relevant_nodes, unmatched_nodes, dataset, node_embeddings):
    """
    抽出したノードと類似ノードに基づいて、商品名ごとの重みの合計を計算
    """
    edge_data = dataset["edges"]
    product_scores = {}

    # 抽出されたノードのスコア計算
    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]
        
        if source in relevant_nodes or target in relevant_nodes:
            product_name = source if source not in relevant_nodes else target
            if product_name not in product_scores:
                product_scores[product_name] = 0.0
            product_scores[product_name] += weight

    # 抽出されなかった単語の類似スコアを計算して加算
    for unmatched_lemma, lemma_vector in unmatched_nodes:
        max_similarity = 0.0
        best_match_node = None

        # 最類似ノードを探す
        for node, node_vector in node_embeddings.items():
            similarity = compute_similarity(lemma_vector, node_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_node = node

        # 最類似ノードと商品名ノード間のスコアを加算
        if best_match_node:
            for edge in edge_data:
                if edge["source"] == best_match_node or edge["target"] == best_match_node:
                    product_name = edge["source"] if edge["source"] != best_match_node else edge["target"]
                    weight = edge["weight"]

                    # スコアに「類似度 × エッジの重み」を加算
                    if product_name not in product_scores:
                        product_scores[product_name] = 0.0
                    product_scores[product_name] += max_similarity * weight

    return product_scores

def calculate_total_weights_with_similarity(
    relevant_nodes, unmatched_nodes, dataset, node_embeddings,
    similarity_threshold=0.3, score_threshold=0.5
):
    """
    抽出したノードと類似ノードに基づいて、商品名ごとの重みの合計を計算
    閾値を導入：
    1. similarity_threshold: 近似ノードとして認める類似度の閾値
    2. score_threshold: 最終的なスコアの閾値
    """
    edge_data = dataset["edges"]
    product_scores = {}

    # 一致したノードのスコア計算（直接マッチ）
    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        if source in relevant_nodes or target in relevant_nodes:
            product_name = source if source not in relevant_nodes else target
            if product_name not in product_scores:
                product_scores[product_name] = 0.0
            product_scores[product_name] += weight

    # 一致しなかった単語の類似スコアを計算
    for unmatched_lemma, lemma_vector in unmatched_nodes:
        max_similarity = 0.0
        best_match_node = None

        # 最類似ノードを探す（類似度が similarity_threshold を超えたもののみ採用）
        for node, node_vector in node_embeddings.items():
            similarity = compute_similarity(lemma_vector, node_vector)
            if similarity > max_similarity and similarity >= similarity_threshold:
                max_similarity = similarity
                best_match_node = node

        # 最類似ノードが見つかった場合のみスコア加算
        if best_match_node:
            for edge in edge_data:
                if edge["source"] == best_match_node or edge["target"] == best_match_node:
                    product_name = edge["source"] if edge["source"] != best_match_node else edge["target"]
                    weight = edge["weight"]

                    # スコアに「類似度 × エッジの重み」を加算
                    if product_name not in product_scores:
                        product_scores[product_name] = 0.0
                    product_scores[product_name] += max_similarity * weight

    # score_threshold を超える商品のみ結果として残す
    filtered_scores = {k: v for k, v in product_scores.items() if v >= score_threshold}

    return filtered_scores


# def calculate_total_weights_with_keyword_search(relevant_nodes, dataset):
#     """
#     一致したノードのみを対象に商品名ごとの重みの合計を計算するキーワード検索処理
#     """
#     edge_data = dataset["edges"]
#     product_scores = {}

#     # 抽出されたノードのスコア計算
#     for edge in edge_data:
#         source = edge["source"]
#         target = edge["target"]
#         weight = edge["weight"]

#         # relevant_nodes に一致する場合のみスコアを計算
#         if source in relevant_nodes or target in relevant_nodes:
#             product_name = source if source not in relevant_nodes else target
#             if product_name not in product_scores:
#                 product_scores[product_name] = 0.0
#             product_scores[product_name] += weight

#     return product_scores

def draw_knowledge_graph_with_similarity(
    relevant_nodes, unmatched_nodes, best_match, dataset, node_embeddings
):
    """
    ナレッジグラフを描画する。
    - 検索結果の商品名ノードのみを描画。
    - 抽出されたノード、最類似ノードを含む。
    - 類似ノードと商品名ノード間にもエッジを追加。
    """
    G = nx.Graph()
    edge_data = dataset["edges"]
    unmatched_edges = []

    # 検索結果の商品名ノードを追加
    G.add_node(best_match, color="green")

    # 抽出されたノードと商品名ノードをつなぐエッジを追加
    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        if (source == best_match and target in relevant_nodes) or (target == best_match and source in relevant_nodes):
            G.add_edge(source, target, weight=weight)

    # 抽出されなかったノードと最類似ノードを追加し、商品名ノードともエッジをつなぐ
    for unmatched_lemma, lemma_vector in unmatched_nodes:
        max_similarity = 0.0
        best_match_node = None

        # 最類似ノードを見つける
        for node, node_vector in node_embeddings.items():
            similarity = compute_similarity(lemma_vector, node_vector)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_node = node

        # 最類似ノードが見つかった場合にグラフに追加
        if best_match_node:
            G.add_node(unmatched_lemma, color="red")  # 抽出されなかったノード
            G.add_node(best_match_node, color="blue")  # 最類似ノード
            G.add_edge(unmatched_lemma, best_match_node, weight=max_similarity)  # 類似度で接続

            # 商品名ノードと類似ノードを接続
            for edge in edge_data:
                if (edge["source"] == best_match_node and edge["target"] == best_match) or \
                   (edge["target"] == best_match_node and edge["source"] == best_match):
                    G.add_edge(best_match_node, best_match, weight=edge["weight"])

    # グラフ描画
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(15, 10))

    # ノードの色分け
    node_colors = [G.nodes[node].get("color", "lightblue") for node in G.nodes]

    # ノードとエッジの描画
    nx.draw(
        G, pos, with_labels=True, node_size=3000,
        node_color=node_colors, font_size=10, font_family='Meiryo'
    )

    # エッジラベル（重み）を描画
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='Meiryo')

    plt.title('ナレッジグラフ（検索結果と最類似ノード）')
    plt.show()

def find_top_matches_with_similarity(input_text, dataset_file, threshold=0.5, top_n=3):
    """
    近似検索を含む商品名の検索（上位N件を返す）
    """
    # データセットをロード
    dataset = load_dataset(dataset_file)
    node_embeddings = {node["node"]: np.array(node["embedding"]) for node in dataset["nodes"]}
    dataset_nodes = set(node_embeddings.keys())
    
    # 抽出ノードと未一致ノードを取得
    relevant_nodes, unmatched_nodes = extract_relevant_nodes_with_similarity(
        input_text, dataset_nodes, node_embeddings, threshold
    )
    if not relevant_nodes and not unmatched_nodes:
        return "一致するノードが見つかりませんでした。"

    # スコア計算
    product_scores = calculate_total_weights_with_similarity(
        relevant_nodes, unmatched_nodes, dataset, node_embeddings
    )
    if not product_scores:
        return "一致する商品が見つかりませんでした。"

    # スコアが高い順にソートして上位N件を取得
    sorted_scores = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    top_matches = sorted_scores[:top_n]  # 上位N件を取得

    return top_matches, relevant_nodes, unmatched_nodes


def find_best_match_with_similarity(input_text, dataset_file, threshold=0.5):
    """
    近似検索を含む商品名の検索
    """
    # データセットをロード
    dataset = load_dataset(dataset_file)
    node_embeddings = {node["node"]: np.array(node["embedding"]) for node in dataset["nodes"]}
    dataset_nodes = set(node_embeddings.keys())
    
    # 抽出ノードと未一致ノードを取得
    relevant_nodes, unmatched_nodes = extract_relevant_nodes_with_similarity(
        input_text, dataset_nodes, node_embeddings, threshold
    )
    if not relevant_nodes and not unmatched_nodes:
        return "一致するノードが見つかりませんでした。"

    # スコア計算
    product_scores = calculate_total_weights_with_similarity(
        relevant_nodes, unmatched_nodes, dataset, node_embeddings
    )
    if not product_scores:
        return "一致する商品が見つかりませんでした。"

    # 最もスコアの高い商品を返す
    best_match = max(product_scores, key=product_scores.get)
    return best_match, product_scores[best_match], relevant_nodes, unmatched_nodes

def append_to_log_file(log_file, query, result):
    """
    検索クエリと結果をログファイルに追記
    """
    log_data = []
    
    # 既存のログファイルを読み込む（存在する場合）
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            try:
                log_data = json.load(f)
            except json.JSONDecodeError:
                log_data = []

    # 新しいエントリを追加
    log_data.append({"検索クエリ": query, "出力結果": result})

    # ログファイルに書き込み
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

    print(f"検索クエリと結果が {log_file} に保存されました。")

if __name__ == "__main__":
    dataset_file = "output/embeddings_direct2.2_entity2.0_similarity0.5.json"
    log_file = "search_log.json"

    input_text = input("商品を探したい内容を入力してください: ")

    # 商品名検索（上位3件を表示）
    result = find_top_matches_with_similarity(input_text, dataset_file, top_n=3)
    if isinstance(result, str):
        print(result)
    else:
        top_matches, relevant_nodes, unmatched_nodes = result

        # 結果を表示
        print("検索結果（上位3件）:")
        for rank, (product_name, score) in enumerate(top_matches, start=1):
            print(f"{rank}. {product_name} (スコア: {score:.4f})")

        # 結果をログファイルに追記
        append_to_log_file(log_file, input_text, {
            "結果": [{product_name: score} for product_name, score in top_matches]
        })

        # ナレッジグラフを描画
        dataset = load_dataset(dataset_file)
        node_embeddings = {node["node"]: np.array(node["embedding"]) for node in dataset["nodes"]}
        best_match = top_matches[0][0]  # 最上位の結果をナレッジグラフに使用
        draw_knowledge_graph_with_similarity(
            relevant_nodes, unmatched_nodes, best_match, dataset, node_embeddings
        )

"""１位検索
# 使用例
if __name__ == "__main__":
    dataset_file = "output/graph_data10.json"
    log_file = "search_log.json"

    input_text = input("商品を探したい内容を入力してください: ")

    # 商品名検索
    result = find_best_match_with_similarity(input_text, dataset_file)
    if isinstance(result, str):
        print(result)
    else:
        best_match, score, relevant_nodes, unmatched_nodes = result
        # 結果をログファイルに追記
        append_to_log_file(log_file, input_text, {"商品名": best_match, "スコア": score})
        print(f"検索結果: {best_match} (スコア: {score})")

        # ナレッジグラフを描画
        dataset = load_dataset(dataset_file)
        node_embeddings = {node["node"]: np.array(node["embedding"]) for node in dataset["nodes"]}
        draw_knowledge_graph_with_similarity(
            relevant_nodes, unmatched_nodes, best_match, dataset, node_embeddings
        )
"""