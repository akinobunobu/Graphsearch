import json
import time
import os
from typing import List, Tuple
import numpy as np

# 評価対象の関数（search_systemは検索システムのメイン関数に置き換えてください）
from mainmain.search import find_top_matches_with_similarity
from mainmain.search_keyword import find_top_matches
from mainmain.search_vec import vector_search

def evaluate_system_search(
    evaluation_file: str,
    dataset_file: str,
    threshold: float = 0.5,
    top_n: int = 3
) -> dict:
    """
    検索システムを評価するスクリプト。
    - 上位1位以内、2位以内、3位以内のPrecisionを計算。

    Args:
        evaluation_file (str): 評価用JSONファイルのパス。
        dataset_file (str): データセットファイルのパス。
        threshold (float): 類似度の閾値。
        top_n (int): 上位N位までの精度を評価。

    Returns:
        dict: 各上位N位以内のPrecisionと平均検索時間。
    """
    # 評価データを読み込む
    if not os.path.exists(evaluation_file):
        raise FileNotFoundError(f"評価用ファイル '{evaluation_file}' が見つかりません。")

    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            if os.stat(evaluation_file).st_size == 0:
                raise ValueError(f"評価用ファイル '{evaluation_file}' が空です。")
            evaluation_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"評価用ファイル '{evaluation_file}' のJSONデータが無効です: {e}")

    # 各上位N位以内の正解数と検索時間
    true_positive = [0] * top_n  # 上位1位, 2位, ..., N位の正解数
    total_time = 0  # 総検索時間

    print(f"\n評価に使用する埋め込みファイル: {dataset_file}")

    # 各クエリで評価
    for data in evaluation_data:
        query = data["search_text"]
        expected_product_name = data["tool_name"]

        # 検索実行と時間計測
        start_time = time.time()
        result = find_top_matches_with_similarity(query, dataset_file, top_n=top_n)
        end_time = time.time()
        total_time += (end_time - start_time)

        # 結果の判定
        if isinstance(result, str):  # 結果が見つからなかった場合
            continue
        else:
            top_matches, _, _ = result
            top_match_names = [product_name for product_name, _ in top_matches]

            # 上位N位以内に含まれるかを判定
            for i in range(top_n):
                if expected_product_name in top_match_names[:i + 1]:
                    true_positive[i] += 1

    # Precisionを計算
    total_queries = len(evaluation_data)
    precision = {
        f"Top-{i + 1} Precision": true_positive[i] / total_queries if total_queries > 0 else 0
        for i in range(top_n)
    }
    average_search_time = total_time / total_queries if total_queries > 0 else 0

    precision["Average Search Time"] = average_search_time
    return precision

def evaluate_system_vec(
    evaluation_file: str,
    graph_data_file: str,
    top_n: int = 3
) -> dict:
    """
    ベクトル検索システムを評価するスクリプト。
    - 上位1位以内、2位以内、3位以内のPrecisionを計算。

    Args:
        evaluation_file (str): 評価用JSONファイルのパス。
        graph_data_file (str): ベクトル埋め込みが含まれるグラフデータのファイルパス。
        top_n (int): 上位N位までの精度を評価。

    Returns:
        dict: 各上位N位以内のPrecisionと平均検索時間。
    """
    # 評価データを読み込む
    if not os.path.exists(evaluation_file):
        raise FileNotFoundError(f"評価用ファイル '{evaluation_file}' が見つかりません。")

    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            if os.stat(evaluation_file).st_size == 0:
                raise ValueError(f"評価用ファイル '{evaluation_file}' が空です。")
            evaluation_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"評価用ファイル '{evaluation_file}' のJSONデータが無効です: {e}")

    # グラフデータを読み込む
    with open(graph_data_file, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # 商品の埋め込みを準備
    product_embeddings = {
        node["node"]: np.array(node["embedding"]) for node in graph_data["nodes"] if "embedding" in node
    }

    # 各上位N位以内の正解数と検索時間
    true_positive = [0] * top_n  # 上位1位, 2位, ..., N位の正解数
    total_time = 0  # 総検索時間

    print(f"\n評価に使用するグラフデータファイル: {graph_data_file}")

    # 各クエリで評価
    for data in evaluation_data:
        query = data["search_text"]
        expected_product_name = data["tool_name"]

        # 検索実行と時間計測
        start_time = time.time()
        vector_results = vector_search(query, graph_data_file, top_n=top_n)
        end_time = time.time()
        total_time += (end_time - start_time)

        # 結果の判定
        top_match_names = [product_name for product_name, _ in vector_results]

        # 上位N位以内に含まれるかを判定
        for i in range(top_n):
            if expected_product_name in top_match_names[:i + 1]:
                true_positive[i] += 1

    # Precisionを計算
    total_queries = len(evaluation_data)
    precision = {
        f"Top-{i + 1} Precision": true_positive[i] / total_queries if total_queries > 0 else 0
        for i in range(top_n)
    }
    average_search_time = total_time / total_queries if total_queries > 0 else 0

    precision["Average Search Time"] = average_search_time
    return precision

def evaluate_system_keyword(
    evaluation_file: str,
    dataset_file: str,
    top_n: int = 3
) -> dict:
    """
    検索システムを評価するスクリプト。
    - 上位1位以内、2位以内、3位以内のPrecisionを計算。

    Args:
        evaluation_file (str): 評価用JSONファイルのパス。
        dataset_file (str): データセットファイルのパス。
        top_n (int): 上位N位までの精度を評価。

    Returns:
        dict: 各上位N位以内のPrecisionと平均検索時間。
    """
    # 評価データを読み込む
    if not os.path.exists(evaluation_file):
        raise FileNotFoundError(f"評価用ファイル '{evaluation_file}' が見つかりません。")

    try:
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            if os.stat(evaluation_file).st_size == 0:
                raise ValueError(f"評価用ファイル '{evaluation_file}' が空です。")
            evaluation_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"評価用ファイル '{evaluation_file}' のJSONデータが無効です: {e}")

    # 各上位N位以内の正解数と検索時間
    true_positive = [0] * top_n  # 上位1位, 2位, ..., N位の正解数
    total_time = 0  # 総検索時間

    print(f"\n評価に使用するデータセットファイル: {dataset_file}")

    # 各クエリで評価
    for data in evaluation_data:
        query = data["search_text"]
        expected_product_name = data["tool_name"]

        # 検索実行と時間計測
        start_time = time.time()
        result = find_top_matches(query, dataset_file, top_n=top_n)
        end_time = time.time()
        total_time += (end_time - start_time)

        # 結果の判定
        if isinstance(result, str):  # 結果が見つからなかった場合
            continue
        else:
            top_matches, relevant_nodes = result
            top_match_names = [product_name for product_name, _ in top_matches]

            # 上位N位以内に含まれるかを判定
            for i in range(top_n):
                if expected_product_name in top_match_names[:i + 1]:
                    true_positive[i] += 1

    # Precisionを計算
    total_queries = len(evaluation_data)
    precision = {
        f"Top-{i + 1} Precision": true_positive[i] / total_queries if total_queries > 0 else 0
        for i in range(top_n)
    }
    average_search_time = total_time / total_queries if total_queries > 0 else 0

    precision["Average Search Time"] = average_search_time
    return precision

if __name__ == "__main__":
    evaluation_file = "evaluate/queli/70tools_queli.json"
    dataset_file = "mainmain/output/embeddings_direct2.2_entity2.0_similarity0.5.json"

    try:
        results = evaluate_system_search(evaluation_file, dataset_file, top_n=3)
        for key, value in results.items():
            print(f"グラフ検索_{key}: {value:.2f}")
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")

    # try:
    #     results = evaluate_system_vec(evaluation_file, dataset_file, top_n=3)
    #     for key, value in results.items():
    #         print(f"ベクトル検索_{key}: {value:.2f}")
    # except (FileNotFoundError, ValueError) as e:
    #     print(f"エラー: {e}")

    # try:
    #     results = evaluate_system_keyword(evaluation_file, dataset_file, top_n=3)
    #     for key, value in results.items():
    #         print(f"キーワード検索_{key}: {value:.2f}")
    # except (FileNotFoundError, ValueError) as e:
    #     print(f"エラー: {e}")
