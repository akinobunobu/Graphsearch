import spacy
import json
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


def extract_relevant_nodes(input_text, dataset_nodes):
    """
    入力テキスト内の動詞、名詞、形容詞を抽出し、
    商品データセット内のノードに関連付ける
    """
    doc = nlp(input_text)
    relevant_nodes = set()

    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "ADJ"]:
            lemma = token.lemma_
            if lemma in dataset_nodes:
                relevant_nodes.add(lemma)

    return list(relevant_nodes)


def calculate_total_weights(relevant_nodes, dataset):
    """
    抽出したノードに基づいて、商品名ごとの重みの合計を計算
    """
    edge_data = dataset["edges"]
    product_scores = {}

    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        if source in relevant_nodes or target in relevant_nodes:
            product_name = source if source not in relevant_nodes else target
            if product_name not in product_scores:
                product_scores[product_name] = 0.0
            product_scores[product_name] += weight

    return product_scores


def draw_knowledge_graph(best_match, relevant_nodes, dataset):
    """
    ナレッジグラフを描画する。
    - 最もスコアの高い商品名ノードを描画。
    - 抽出されたノードを強調。
    """
    G = nx.Graph()
    edge_data = dataset["edges"]

    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        weight = edge["weight"]

        if (source == best_match and target in relevant_nodes) or (target == best_match and source in relevant_nodes):
            G.add_edge(source, target, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(15, 10))

    # ノードの色分け
    node_colors = ["green" if node == best_match else "lightblue" for node in G.nodes]

    # ノードとエッジの描画
    nx.draw(
        G, pos, with_labels=True, node_size=3000,
        node_color=node_colors, font_size=10, font_family='Meiryo'
    )

    # エッジラベル（重み）を描画
    edge_labels = nx.get_edge_attributes(G, 'weight')
    edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='Meiryo')

    plt.title(f'ナレッジグラフ（最も関連性の高い商品: {best_match}）')
    plt.show()


def find_top_matches(input_text, dataset_file, top_n=3):
    """
    入力テキストに基づいて商品名を検索（上位N件を返す）
    """
    dataset = load_dataset(dataset_file)
    dataset_nodes = {node["node"] for node in dataset["nodes"]}

    # 抽出ノードを取得
    relevant_nodes = extract_relevant_nodes(input_text, dataset_nodes)
    if not relevant_nodes:
        return "一致するノードが見つかりませんでした。"

    # スコア計算
    product_scores = calculate_total_weights(relevant_nodes, dataset)
    if not product_scores:
        return "一致する商品が見つかりませんでした。"

    # スコアが高い順にソートして上位N件を取得
    sorted_scores = sorted(product_scores.items(), key=lambda x: x[1], reverse=True)
    top_matches = sorted_scores[:top_n]

    return top_matches, relevant_nodes


if __name__ == "__main__":
    dataset_file = "output/embeddings_direct2.2_entity2.0_similarity0.5.json"

    input_text = input("商品を探したい内容を入力してください: ")

    # 商品名検索（上位3件を表示）
    result = find_top_matches(input_text, dataset_file, top_n=3)
    if isinstance(result, str):
        print(result)
    else:
        top_matches, relevant_nodes = result

        # 結果を表示
        print("検索結果（上位3件）:")
        for rank, (product_name, score) in enumerate(top_matches, start=1):
            print(f"{rank}. {product_name} (スコア: {score:.4f})")

        # ナレッジグラフを描画（最上位の商品名のみ）
        best_match = top_matches[0][0]  # 最もスコアの高い商品名
        dataset = load_dataset(dataset_file)
        draw_knowledge_graph(best_match, relevant_nodes, dataset)
