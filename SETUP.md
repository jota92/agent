# AI Browser Agent - セットアップガイド

## 🎯 概要

ローカルの Transformer ベース視覚言語モデルを使用してウェブサイトを自動操作する AI エージェントシステムです。\
Google 検索などの一般的なブラウジングタスクは自動的にサブステップへ分解され、UI 左側に「Agent Plan」として可視化されます。

## 📋 必要な環境

- Linux または macOS
- Python 3.8 以上（3.10 推奨）
- Google Chrome / Chromium
- ChromeDriver（ブラウザとバージョンを揃える）
- 16GB 以上のメモリ推奨（モデル展開で ~8GB 使用）
- 初回のみモデルダウンロード用のインターネット接続（約 4GB）

## 🚀 セットアップ手順

### 1. リポジトリの取得

```bash
git clone https://github.com/YOUR_USERNAME/agent.git
cd agent
```

### 2. クイックスタートスクリプトを実行

```bash
chmod +x start.sh
./start.sh
```

スクリプトは以下の処理を自動化します:

- Python 仮想環境の作成と有効化
- 必要な Python パッケージのインストール
- Qwen2-VL-2B Instruct モデルのダウンロード（初回のみ約 4GB）
- サーバーの起動

初回ダウンロードが完了するまで数分かかります。ログに「Vision-language model ready」と表示されるまでお待ちください。

### 3. ChromeDriver の確認

```bash
which chromedriver
```

パスが表示されない場合は、使用環境に合わせて ChromeDriver をインストールしてください。Linux では公式サイトからバイナリを配置するか、パッケージマネージャーを利用します。macOS では `brew install chromedriver` が利用できます。

## ▶️ 起動方法

起動は `./start.sh` を再度実行するだけで OK です。仮想環境とモデルが既に存在する場合は高速に起動します。手動で起動する場合は以下を参考にしてください。

```bash
source venv/bin/activate
python3 app.py
```

サーバーが起動したらブラウザで `http://127.0.0.1:5000` にアクセスします。

## 🎮 使い方

1. **モデル状態確認**: 右上のステータスが「Local VLM: Qwen/Qwen2-VL-2B-Instruct」で緑のドットならモデル読み込み済み

2. **タスクを入力**: 左サイドバーのテキストエリアに自然言語で指示を入力

   例:

   - `Googleで'OpenAI'を検索して、最初の結果をクリックしてください`
   - `YouTubeで'Python tutorial'を検索してください`
   - `Wikipediaで'人工知能'を検索して、記事の冒頭を読んでください`

3. **Start Task ボタンをクリック**: AI が自動的にウェブサイトを操作開始

4. **進行状況を確認**:

   - ブラウザ画面に赤いカーソルが表示され、AI の操作位置を示します
   - 左パネルの "Agent Plan" に構造化ステップと達成状況 (例: 3/5) が表示されます
   - "Task Logs" にリアルタイムで操作ログが表示され、Status セクションに現在のステップ数と状態が表示されます

5. **停止**: "Stop" ボタンでいつでも中断可能

## 🔧 トラブルシューティング

### モデルの読み込みに失敗する

- メモリ不足の可能性があります。不要なアプリを終了してから再実行してください。
- モデルファイルが壊れている場合はキャッシュを削除します。

  ```bash
  rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-2B-Instruct
  ./start.sh
  ```

- プロキシ環境では `HF_ENDPOINT` を設定するなど、Hugging Face へのアクセスを確認してください。

### 初回ダウンロードが非常に遅い

- 4GB 規模のモデルを取得するため、回線状況により時間がかかります。
- `pip install huggingface_hub` 後に `huggingface-cli download Qwen/Qwen2-VL-2B-Instruct` を先に実行しておくと再利用できます。

### 推論が遅い / 軽量化したい

- GPU がある場合は `CUDA_VISIBLE_DEVICES=0 ./start.sh` のように GPU を割り当ててください。
- CPU のみで軽量化するには `app.py` の `VLM_MODEL_ID` を `Qwen/Qwen2-VL-7B-Instruct` や `microsoft/Phi-3.5-vision-instruct` に変更し、[Hugging Face](https://huggingface.co/) から該当モデルを取得してください（メモリ要件に注意）。

### ChromeDriver に関するエラー

- `which chromedriver` でパスを確認し、見つからない場合は環境に応じた手段でインストールしてください。
- バージョン違いの場合はブラウザと同じバージョンを再インストールしてください。
- Linux で `chromedriver` を実行できない場合は実行権限を付与します。

  ```bash
  chmod +x /path/to/chromedriver
  ```

## ⚙️ カスタマイズ

### `app.py` の設定項目

```python
# AI 設定
VLM_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # 使用する視覚言語モデル
AI_MAX_STEPS = 30                 # 最大ステップ数 (増やすと長いタスクに対応)

# パフォーマンス
JPEG_QUALITY = 65                 # 画質 (30-100, 低いほど軽い)
TARGET_STREAM_WIDTH = 1024        # 配信幅 (下げると軽くなる)
ACTIVE_CAPTURE_DELAY = 0.08       # フレーム間隔 (秒)
```

## 📊 推奨モデル比較

| モデル                           | サイズ | 速度 | 精度 | RAM 使用量 | 推奨用途               |
| -------------------------------- | ------ | ---- | ---- | ---------- | ---------------------- |
| Qwen2-VL-2B-Instruct (デフォルト) | ~4GB   | 中   | 高   | 8GB+       | 汎用ブラウジング       |
| Qwen2-VL-7B-Instruct             | ~8GB   | やや遅 | 非常に高 | 16GB+      | 精度重視のタスク       |
| Phi-3.5-vision-instruct          | ~3.5GB | 速   | 中   | 8GB        | 軽量環境・CPU ベース   |

## 🎯 タスク例

### 簡単なタスク

```
Googleのトップページを開いてください
```

### 検索タスク

```
Googleで'Selenium Python'を検索して、最初の3つのリンクを確認してください
```

### 複雑なタスク

```
GitHubで'flask'を検索し、スター数が最も多いリポジトリを見つけてください
```

## 🛡️ 注意事項

- AI は現在の画面を分析して次の操作を決定します
- 複雑なタスクは段階的に分割するとより正確です
- ログインが必要なサイトは事前に手動でログインしてください (セッション維持)
- 最大ステップ数 (デフォルト 30) を超えるとタスクは失敗します

## 📝 ライセンス

このプロジェクトは教育・研究目的で作成されています。商用利用の際は使用する AI モデルのライセンスを確認してください。

---

**🤖 Enjoy your AI Browser Agent!**
