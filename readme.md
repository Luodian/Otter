# alcon2023
author: 柏原功太郎
## インストール&実行
- 仮想環境：`conda env create -f environment.yml`
- 実行：`python main.py --input_csv_path="./input.csv" --output_csv_path="./output.csv"`
  - 実行時には、入力動画のメタデータが書かれたCSVへのパス、出力CSVへのパスを指定してください

## 処理の流れ
※詳細は予稿を確認してください
1. MP4ファイルを読み込み
2. 背景差分を用いて動体以外の背景を透過させる
3. 背景を透過させた動画と、`./prompt`以下にある文章を、動画理解モデル`luodian/OTTER-9B-DenseCaption`に入力する
4. 出力をフォーマットしてCSVに出力する