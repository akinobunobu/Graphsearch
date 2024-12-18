import pandas as pd
import tkinter as tk
from tkinter import ttk
import json

def display_json_as_table(json_file):
    """
    JSONファイルを読み込み、表形式で別ウィンドウに表示
    """
    # JSONファイルを読み込む
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Pandas DataFrameに変換
    df = pd.DataFrame(data)

    # Tkinterウィンドウを作成
    root = tk.Tk()
    root.title("グリッドサーチ結果")

    # スクロール可能なテーブルを作成
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True)

    # ツリービューを作成
    tree = ttk.Treeview(frame, columns=list(df.columns), show="headings", height=25)

    # 各列の設定
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100, anchor="center")

    # データを挿入
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # スクロールバーを作成
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)

    # ツリービューとスクロールバーを配置
    tree.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Tkinterのメインループを開始
    root.mainloop()

# 実行例
if __name__ == "__main__":
    json_file = "grid_search_log.json"  # JSONファイルのパス
    display_json_as_table(json_file)
