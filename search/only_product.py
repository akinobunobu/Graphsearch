import json

# 使用するテキストファイルを指定（例：tool.text のみを使用）
text_file_path = "/mnt/data/tool.text"
embedding_file_path = "/mnt/data/embeddings_direct2.2_entity2.0_similarity0.5.json"
filtered_embedding_file_path = "/mnt/data/filtered_embeddings_tool.json"

# 商品名リストの取得
def load_product_names(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        product_names = {line.strip() for line in f if line.strip()}
    return product_names

# 指定したテキストファイルの商品名を取得
valid_products = load_product_names(text_file_path)

# 埋め込みデータのフィルタリング
with open(embedding_file_path, "r", encoding="utf-8") as f:
    embedding_data = json.load(f)

filtered_nodes = [node for node in embedding_data["nodes"] if node["node"] in valid_products]

# 新しい埋め込みデータを作成
filtered_embedding_data = {"nodes": filtered_nodes}

# フィルタリング後のデータを保存
with open(filtered_embedding_file_path, "w", encoding="utf-8") as f:
    json.dump(filtered_embedding_data, f, ensure_ascii=False, indent=4)

print(f"フィルタリング後の埋め込みデータを保存しました: {filtered_embedding_file_path}")
