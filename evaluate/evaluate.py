import json
import time
import os
from typing import List, Tuple

# 評価対象の関数（search_systemは検索システムのメイン関数に置き換えてください）
from mainmain.search import find_best_match_with_similarity

def evaluate_system(
    evaluation_file: str,
    dataset_file: str,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    検索システムを評価するスクリプト。
    - Precision, 検索速度を計算。

    Args:
        evaluation_file (str): 評価用JSONファイルのパス。
        dataset_file (str): データセットファイルのパス。
        threshold (float): 類似度の閾値。
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

    # 評価結果を格納する変数
    true_positive = 0  # 正解している数
    false_positive = 0  # 間違った提案数
    total_time = 0  # 総検索時間

    print(f"\n評価に使用する埋め込みファイル: {dataset_file}")

    # 各クエリで評価
    for data in evaluation_data:
        query = data["search_text"]
        expected_product_name = data["tool_name"]

        # 検索実行と時間計測
        start_time = time.time()
        result = find_best_match_with_similarity(query, dataset_file, threshold)
        end_time = time.time()
        total_time += (end_time - start_time)

        # 結果の判定
        if isinstance(result, str):  # 結果が見つからなかった場合
            false_negative += 1
        else:
            predicted_product_name, _, _, _ = result
            if predicted_product_name == expected_product_name:
                true_positive += 1
            else:
                false_positive += 1

    # Precision, Recall, F1スコアを計算
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    average_search_time = total_time / len(evaluation_data)

    return precision, average_search_time

# 実行例
if __name__ == "__main__":
    evaluation_file = "evaluate/tools_features.json"
    dataset_file = "mainmain\output\embeddings6.json"

    try:
        precision, avg_time = evaluate_system(evaluation_file, dataset_file)
        print(f"Precision: {precision:.2f}")
        print(f"Average Search Time: {avg_time:.4f} seconds")
    except (FileNotFoundError, ValueError) as e:
        print(f"エラー: {e}")
