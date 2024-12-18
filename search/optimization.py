import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  # 追加

from mainmain.input_direct_by import create_complete_graph_from_text  # グラフ生成関数をインポート
from evaluate.evaluate import evaluate_system  # 評価関数をインポート

def grid_search_for_weights(input_file, output_folder, evaluation_file, log_file, threshold=0.5):
    """
    グリッドサーチを用いて重みの最適化を行い、結果をグラフと表で表示。

    Args:
        input_file (str): 商品データファイルのパス（JSON形式）。
        output_folder (str): グラフデータの出力フォルダ。
        evaluation_file (str): 評価データファイルのパス。
        log_file (str): ログファイルのパス。
        threshold (float): 類似度の閾値。
    
    Returns:
        dict: 最適な重みパラメータと精度。
    """
    # 重みの範囲を0.1刻みで設定
    direct_weights = np.arange(2.4, 3.51, 0.1)  
    entity_weights = np.arange(2.0, 2.01, 0.1)  
    similarity_weights = np.arange(0.7, 0.81, 0.1)  

    best_precision = 0.0
    best_params = None
    log_data = []  # ログデータを格納するリスト

    print("重み設定ごとのPrecision:")
    print("------------------------------------------------")
    print("{:<15} {:<15} {:<15} {:<10}".format("Direct Weight", "Entity Weight", "Similarity Weight", "Precision"))
    print("------------------------------------------------")
    for direct_weight in direct_weights:
        for entity_weight in entity_weights:
            for similarity_weight in similarity_weights:
                # グラフ生成と埋め込みファイル取得
                output_file = create_complete_graph_from_text(
                    input_file=input_file,
                    output_folder=output_folder,
                    direct_weight=direct_weight,
                    entity_weight=entity_weight,
                    similarity_weight=similarity_weight
                )

                # 評価関数に直接指定
                precision, avg_time = evaluate_system(
                    evaluation_file=evaluation_file,
                    dataset_file=output_file,  # 生成した埋め込みファイルを直接指定
                    threshold=threshold
                )

                # 結果をログに記録
                log_data.append({
                    "direct_weight": round(direct_weight, 1),
                    "entity_weight": round(entity_weight, 1),
                    "similarity_weight": round(similarity_weight, 1),
                    "precision": precision,
                    "average_time": avg_time
                })

                # 表形式で結果を表示
                print("{:<15} {:<15} {:<15} {:<10.4f}".format(
                    round(direct_weight, 1),
                    round(entity_weight, 1),
                    round(similarity_weight, 1),
                    precision))

                # 精度が向上した場合に最適な重みを更新
                if precision > best_precision:
                    best_precision = precision
                    best_params = {
                        "direct_weight": round(direct_weight, 1),
                        "entity_weight": round(entity_weight, 1),
                        "similarity_weight": round(similarity_weight, 1),
                        "precision": precision,
                        "average_time": avg_time,
                    }

    print("------------------------------------------------")

    # ログデータをJSONファイルに保存
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

    print(f"\nログが {log_file} に保存されました。")

    # 結果を可視化
    visualize_results(log_data)

    return best_params

def visualize_results(log_data):
    """
    グリッドサーチの結果をグラフと表で可視化。
    """
    # データをPandas DataFrameに変換
    df = pd.DataFrame(log_data)

    # 表の表示
    print("\n結果の概要:")
    print(df)

    # Precisionの分布を棒グラフで表示
    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(df)),
        df["precision"],
        color="blue",
        alpha=0.7
    )
    plt.xticks(
        range(len(df)),
        [f"D:{row['direct_weight']} E:{row['entity_weight']} S:{row['similarity_weight']}" for _, row in df.iterrows()],
        rotation=90
    )
    plt.title("Precision by Weight Combination")
    plt.xlabel("Weight Combination (Direct, Entity, Similarity)")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.show()

# 実行例
if __name__ == "__main__":
    input_file = "mainmain/input/10tools_3descriptions.json"  # 商品データファイル
    output_folder = "output"  # グラフデータの出力フォルダ
    evaluation_file = "evaluate/queli/10tools_queli_new.json"  # 評価データファイル
    log_file = "grid_search_log.json"  # ログファイル

    best_params = grid_search_for_weights(input_file, output_folder, evaluation_file, log_file)

    if best_params:
        print("\n最適なパラメータ:")
        print(json.dumps(best_params, ensure_ascii=False, indent=4))
    else:
        print("最適なパラメータが見つかりませんでした。")
