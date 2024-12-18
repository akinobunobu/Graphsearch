import json
from mainmain.search_system import keyword_search, vector_search

def evaluate_precision_at_k(results, ground_truth, k):
    """
    上位k件以内に正解が含まれるかを判定
    results: 検索結果のリスト [(商品名, スコア), ...]
    ground_truth: 正解の商品名
    k: 判定対象の上位件数
    """
    top_results = [result[0] for result in results[:k]]
    return 1.0 if ground_truth in top_results else 0.0

def evaluate_search_methods(test_file, graph_data_file, top_n=3):
    """
    キーワード検索とベクトル検索の精度を評価
    test_file: テスト用JSONファイル
    graph_data_file: グラフデータのJSONファイル
    top_n: 上位N件までの精度を評価
    """
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # Precisionを格納するリスト
    keyword_precisions_at_k = {k: [] for k in range(1, top_n + 1)}
    vector_precisions_at_k = {k: [] for k in range(1, top_n + 1)}

    for entry in test_data:
        query = entry["search_text"]
        ground_truth = entry["tool_name"]

        # キーワード検索の結果
        keyword_results = keyword_search(query, graph_data_file, top_n)
        for k in range(1, top_n + 1):
            precision = evaluate_precision_at_k(keyword_results, ground_truth, k)
            keyword_precisions_at_k[k].append(precision)

        # ベクトル検索の結果
        vector_results = vector_search(query, graph_data_file, top_n)
        for k in range(1, top_n + 1):
            precision = evaluate_precision_at_k(vector_results, ground_truth, k)
            vector_precisions_at_k[k].append(precision)

    # 各kの平均Precisionを計算
    avg_keyword_precision_at_k = {
        k: sum(keyword_precisions_at_k[k]) / len(keyword_precisions_at_k[k])
        for k in range(1, top_n + 1)
    }
    avg_vector_precision_at_k = {
        k: sum(vector_precisions_at_k[k]) / len(vector_precisions_at_k[k])
        for k in range(1, top_n + 1)
    }

    return avg_keyword_precision_at_k, avg_vector_precision_at_k

if __name__ == "__main__":
    test_file = "evaluate/queli/10tools_queli_new.json"  # テストデータファイル
    graph_data_file = "mainmain/output/embeddings_direct2.2_entity2.0_similarity0.5.json"  # グラフデータファイル
    top_n = 3  # 上位N件

    # 精度評価
    keyword_precisions, vector_precisions = evaluate_search_methods(test_file, graph_data_file, top_n)

    # 結果を出力
    print("\nキーワード検索のPrecision:")
    for k, precision in keyword_precisions.items():
        print(f"Top-{k} Precision: {precision:.2f}")

    print("\nベクトル検索のPrecision:")
    for k, precision in vector_precisions.items():
        print(f"Top-{k} Precision: {precision:.2f}")
