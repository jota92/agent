# AI Browser Agent - セットアップガイド

## 🎯 概要

ローカル AI (Ollama) を使用してウェブサイトを自動操作する AI エージェントシステムです。

## 📋 必要な環境

- macOS (Apple Silicon または Intel)
- Python 3.8 以上
- Google Chrome
- Homebrew

## 🚀 セットアップ手順

### 1. Ollama のインストール

```bash
# Homebrew で Ollama をインストール
brew install ollama

# Ollama サービスを起動
ollama serve
```

別のターミナルウィンドウで:

```bash
# ビジョン対応の LLaVA モデルをダウンロード (推奨: 13B版)
ollama pull llava:13b

# または軽量版 (7B) - メモリが少ない場合
# ollama pull llava:7b

# または小型版 (Moondream) - 更に軽量
# ollama pull moondream
```

### 2. Python 環境のセットアップ

```bash
cd /Users/jota/Downloads/agent

# 仮想環境を作成
python3 -m venv venv

# 仮想環境を有効化
source venv/bin/activate

# 依存パッケージをインストール
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. ChromeDriver の確認

```bash
# ChromeDriver がインストールされているか確認
which chromedriver

# 出力例: /opt/homebrew/bin/chromedriver
# 何も表示されない場合は以下でインストール:
brew install chromedriver
```

## ▶️ 起動方法

### 1. Ollama を起動 (ターミナル 1)

```bash
ollama serve
```

### 2. AI Browser Agent を起動 (ターミナル 2)

```bash
cd /Users/jota/Downloads/agent
source venv/bin/activate
python app.py
```

### 3. ブラウザでアクセス

```
http://127.0.0.1:5000
```

## 🎮 使い方

1. **Ollama 接続確認**: 右上の "Ollama: llava:13b" が緑のドットなら接続成功

2. **タスクを入力**: 左サイドバーのテキストエリアに自然言語で指示を入力

   例:

   - `Googleで'OpenAI'を検索して、最初の結果をクリックしてください`
   - `YouTubeで'Python tutorial'を検索してください`
   - `Wikipediaで'人工知能'を検索して、記事の冒頭を読んでください`

3. **Start Task ボタンをクリック**: AI が自動的にウェブサイトを操作開始

4. **進行状況を確認**:

   - ブラウザ画面に赤いカーソルが表示され、AI の操作位置を示します
   - 左パネルの "Task Logs" にリアルタイムで操作ログが表示されます
   - Status セクションに現在のステップ数と状態が表示されます

5. **停止**: "Stop" ボタンでいつでも中断可能

## 🔧 トラブルシューティング

### Ollama に接続できない

```bash
# Ollama が起動しているか確認
ps aux | grep ollama

# 起動していない場合
ollama serve

# ポート 11434 が使用中か確認
lsof -i :11434
```

### llava:13b モデルがない

```bash
# 利用可能なモデルを確認
ollama list

# llava がない場合は再ダウンロード
ollama pull llava:13b
```

### AI の応答が遅い・精度が低い

- **llava:7b** を試す (軽量版):

  ```bash
  ollama pull llava:7b
  ```

  その後、`app.py` の 26 行目を編集:

  ```python
  OLLAMA_MODEL = "llava:7b"
  ```

- **moondream** を試す (超軽量):
  ```bash
  ollama pull moondream
  ```
  `app.py` を編集:
  ```python
  OLLAMA_MODEL = "moondream"
  ```

### ChromeDriver エラー

```bash
# ChromeDriver のパスを確認
which chromedriver

# Homebrew で再インストール
brew reinstall chromedriver

# セキュリティ警告が出た場合
xattr -d com.apple.quarantine /opt/homebrew/bin/chromedriver
```

## ⚙️ カスタマイズ

### `app.py` の設定項目

```python
# AI 設定
OLLAMA_MODEL = "llava:13b"       # 使用する AI モデル
AI_MAX_STEPS = 30                 # 最大ステップ数 (増やすと長いタスクに対応)

# パフォーマンス
JPEG_QUALITY = 65                 # 画質 (30-100, 低いほど軽い)
TARGET_STREAM_WIDTH = 1024        # 配信幅 (下げると軽くなる)
ACTIVE_CAPTURE_DELAY = 0.08       # フレーム間隔 (秒)
```

## 📊 推奨モデル比較

| モデル    | サイズ | 速度 | 精度 | RAM 使用量 | 推奨用途         |
| --------- | ------ | ---- | ---- | ---------- | ---------------- |
| llava:13b | ~8GB   | 中   | 高   | 16GB+      | 本番・高精度     |
| llava:7b  | ~4GB   | 速   | 中   | 8GB+       | テスト・バランス |
| moondream | ~2GB   | 爆速 | 低   | 4GB+       | 開発・動作確認   |

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
