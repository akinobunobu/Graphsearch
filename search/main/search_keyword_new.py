import spacy
import json
import networkx as nx
import matplotlib.pyplot as plt

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

def calculate_match_count(relevant_nodes, dataset):
    """
    抽出したノードに基づいて、商品名ごとの一致回数をカウント
    （意味的類似度は考慮せず、単純な一致回数をスコアとする）
    """
    edge_data = dataset["edges"]
    product_scores = {}

    for edge in edge_data:
        source = edge["source"]
        target = edge["target"]
        
        if source in relevant_nodes or target in relevant_nodes:
            product_name = source if source not in relevant_nodes else target
            if product_name not in product_scores:
                product_scores[product_name] = 0
            product_scores[product_name] += 1  # 一致ごとに1加算

    return product_scores

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

    # スコア計算（単純な一致回数ベース）
    product_scores = calculate_match_count(relevant_nodes, dataset)
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
            print(f"{rank}. {product_name} (スコア: {score})")
