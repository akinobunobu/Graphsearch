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
    """
    グラフ内のエッジ重みを最適化する関数。
    """
    for u, v, data in G.edges(data=True):
        # エッジが固有表現ノードに関連している場合
        if named_entity_nodes and (u in named_entity_nodes or v in named_entity_nodes):
            G[u][v]['weight'] *= entity_weight
        else:
            # コサイン類似度の場合は similarity_weight を適用
            if data['weight'] != 1.0:
                G[u][v]['weight'] *= similarity_weight
            else:
                # 直接関連性（固定重み1.0）には direct_weight を適用
                G[u][v]['weight'] *= direct_weight

def create_complete_graph_from_text(input_file, output_folder, direct_weight=1.8, entity_weight=2.0, similarity_weight=1.0):
    """
    JSONファイルから商品説明を読み込み、無向グラフを構築。
    固有表現ノードにかかわるエッジの重みを2倍に設定。
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
        
        # 商品名ノードを追加
        if product_name not in G.nodes:
            G.add_node(product_name)
            node_embeddings[product_name] = nlp(product_name).vector

        # 各商品説明に含まれるノードを抽出
        description_nodes = set()
        for text in descriptions:
            doc = nlp(text)
            for token in doc:
                if token.pos_ in ["NOUN", "VERB", "ADJ"]:
                    lemma = token.lemma_  # 単語を原形化
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
                named_entity_nodes.add(ent.text)  # 固有表現ノードを記録

        # 商品名ノードと商品説明内ノードを接続（
        for node in description_nodes:
            G.add_edge(product_name, node, weight=1.0)

    # 商品名ノードと他のノード間のエッジを追加（コサイン類似度を使用）
    nodes = list(G.nodes)
    for product_name in node_embeddings.keys():
        if product_name in [p["product_name"] for p in product_data]:  # 商品名ノード
            for node in nodes:
                if node != product_name:  # 自己ループを回避
                    vec1 = node_embeddings[product_name]
                    vec2 = node_embeddings.get(node)
                    if vec2 is not None:
                        similarity = compute_similarity(vec1, vec2)
                        if not G.has_edge(product_name, node):  # 重複エッジを防止
                            G.add_edge(product_name, node, weight=float(similarity))

    # エッジの重みを最適化
        optimize_edge_weights(
            G,
            direct_weight=direct_weight,
            entity_weight=entity_weight,
            similarity_weight=similarity_weight,
            named_entity_nodes=named_entity_nodes
        )

    # # 固有表現ノードに関係するエッジの重みを2倍に設定
    # for u, v, data in G.edges(data=True):
    #     if u in named_entity_nodes or v in named_entity_nodes:
    #         G[u][v]['weight'] *= 2.0

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

    # # 新しい出力ファイル名を生成
    # existing_files = [f for f in os.listdir(output_folder) if f.startswith("embeddings") and f.endswith(".json")]
    # new_index = len(existing_files) + 1
    # output_file = os.path.join(output_folder, f"embeddings{new_index}.json")

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
    # print(f"ノードとエッジ情報が {output_file} に保存されました。")

    # # グラフの描画
    # pos = nx.spring_layout(G, seed=42)
    # plt.figure(figsize=(15, 10))
    # nx.draw(
    #     G, pos, with_labels=True, node_size=3000, 
    #     node_color='lightblue', font_size=10, font_family='Meiryo'
    # )
    # # エッジラベルを類似度として描画
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # edge_labels = {k: f"{v:.2f}" for k, v in edge_labels.items()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_family='Meiryo')

    # plt.title('無向グラフ')
    # plt.show()

# 使用例
if __name__ == "__main__":
    input_file = 'input/10tools_simple.json'  # 商品説明の入力ファイル
    output_folder = 'output'  # 出力フォルダ
    create_complete_graph_from_text(input_file, output_folder)
