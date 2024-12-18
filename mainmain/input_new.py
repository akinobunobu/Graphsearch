import spacy
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import json
import os
import gc

# 日本語フォントの設定
rcParams['font.family'] = 'Meiryo'

# SpaCyの日本語モデルのロード
nlp = spacy.load("ja_core_news_lg")

def compute_similarity(vec1, vec2):
    """2つのベクトル間の類似度（コサイン類似度）を計算"""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def optimize_edge_weights(G, direct_weight, entity_weight, similarity_weight, named_entity_nodes=None):
    """グラフ内のエッジ重みを最適化する関数"""
    for u, v, data in G.edges(data=True):
        if named_entity_nodes and (u in named_entity_nodes or v in named_entity_nodes):
            G[u][v]['weight'] *= entity_weight
        else:
            if data['weight'] != 1.0:
                G[u][v]['weight'] *= similarity_weight
            else:
                G[u][v]['weight'] *= direct_weight

def create_complete_graph_from_text(input_file, output_folder, direct_weight=1.8, entity_weight=2.0, similarity_weight=1.0):
    """
    JSONファイルから商品説明を読み込み、無向グラフを構築。
    商品名の埋め込みを商品説明全体の埋め込みに基づいて設定。
    """
    # JSONファイルを読み込む
    with open(input_file, 'r', encoding='utf-8') as f:
        product_data = json.load(f)
    
    # グラフ作成
    G = nx.Graph()
    node_embeddings = {}
    named_entity_nodes = set()  # 固有表現ノードを記録

    # 各商品の埋め込みを取得
    for product in product_data:
        product_name = product['product_name']  # 商品名
        descriptions = product['descriptions']  # 商品説明リスト

        # 商品説明全体の埋め込みを計算
        description_vectors = []
        for text in descriptions:
            doc = nlp(text)
            description_vectors.append(doc.vector)  # 各文のベクトルを取得

        # 商品説明全体の平均ベクトルを計算
        if description_vectors:
            product_embedding = np.mean(description_vectors, axis=0)
        else:
            product_embedding = np.zeros(nlp("dummy").vector.shape)

        # 商品名ノードを追加し、埋め込みを設定
        if product_name not in G.nodes:
            G.add_node(product_name)
            node_embeddings[product_name] = product_embedding

        # 各商品説明に含まれるノードを抽出
        description_nodes = set()
        for text in descriptions:
            doc = nlp(text)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                    lemma = token.lemma_
                    if lemma not in G.nodes:
                        G.add_node(lemma)
                        node_embeddings[lemma] = nlp(lemma).vector
                    description_nodes.add(lemma)

            # 固有表現を抽出して記録
            for ent in doc.ents:
                if ent.text not in G.nodes:
                    G.add_node(ent.text)
                    node_embeddings[ent.text] = nlp(ent.text).vector
                description_nodes.add(ent.text)
                named_entity_nodes.add(ent.text)

        # 商品名ノードと商品説明内ノードを接続
        for node in description_nodes:
            G.add_edge(product_name, node, weight=1.0)

    # 商品名ノードと他のノード間のエッジを追加（コサイン類似度を使用）
    nodes = list(G.nodes)
    for product_name in node_embeddings.keys():
        if product_name in [p["product_name"] for p in product_data]:  # 商品名ノード
            for node in nodes:
                if node != product_name:
                    vec1 = node_embeddings[product_name]
                    vec2 = node_embeddings.get(node)
                    if vec2 is not None:
                        similarity = compute_similarity(vec1, vec2)
                        if not G.has_edge(product_name, node):
                            G.add_edge(product_name, node, weight=float(similarity))

    # エッジの重みを最適化
    optimize_edge_weights(
        G,
        direct_weight=direct_weight,
        entity_weight=entity_weight,
        similarity_weight=similarity_weight,
        named_entity_nodes=named_entity_nodes
    )

    # フォルダが存在しない場合は作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # グラフデータをJSON形式で保存
    nodes_data = [{"node": node, "embedding": [float(x) for x in node_embeddings.get(node, [])]} for node in G.nodes]
    edges_data = [{"source": u, "target": v, "weight": float(d["weight"])} for u, v, d in G.edges(data=True)]

    graph_output = {
        "nodes": nodes_data,
        "edges": edges_data
    }

    output_file = os.path.join(
        output_folder,
        f"embeddings_direct{direct_weight:.1f}_entity{entity_weight:.1f}_similarity{similarity_weight:.1f}.json"
    )

    # 埋め込みデータをJSONファイルに保存
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(graph_output, f, ensure_ascii=False, indent=4)

    # ディスクへの書き込みをフラッシュ
    if hasattr(os, "sync"):
        os.sync()
    print(f"Generated graph file: {output_file}")

    # キャッシュ削除
    gc.collect()

    return output_file

# 使用例
if __name__ == "__main__":
    input_file = 'input/10tools.json'  # 商品説明の入力ファイル
    output_folder = 'output'  # 出力フォルダ
    create_complete_graph_from_text(input_file, output_folder)
