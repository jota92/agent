# AI Browser Agent 🤖

AIがブラウザを自動操作するローカルエージェントシステム。視覚認識モデルを使用してウェブページを理解し、マウスカーソルを動かして実際にクリックやタイピングを実行します。

## 特徴 ✨

- **🎯 視覚的AI操作**: Llama 3.2 Vision モデルでスクリーンショットを解析
- **🖱️ リアルタイムカーソル**: AIが操作する巨大なカーソルをブラウザ上に表示
- **📊 座標グリッド**: 100px間隔のグリッドで正確な座標指定
- **🔄 スムーズアニメーション**: カーソルが滑らかに移動（20ステップ、smoothstep）
- **🛡️ ループ防止**: トリプル防止システムで無限ループを回避
- **🎨 ライブストリーミング**: ブラウザの状態をリアルタイム表示

## 必要要件 📋

- macOS (Apple Silicon推奨)
- Python 3.8+
- Ollama
- ChromeDriver

## セットアップ 🚀

1. **リポジトリをクローン**
```bash
git clone https://github.com/YOUR_USERNAME/agent.git
cd agent
```

2. **クイックスタート**
```bash
chmod +x start.sh
./start.sh
```

このスクリプトが自動的に:
- Ollamaのインストール・起動
- Llama 3.2 Vision モデルのダウンロード
- Python仮想環境のセットアップ
- 依存関係のインストール
- サーバーの起動

3. **ブラウザでアクセス**
```
http://127.0.0.1:5000
```

## 使い方 💡

1. ブラウザで http://127.0.0.1:5000 を開く
2. URLを入力してブラウザを開く
3. タスクを入力（例: "googleで'openai'を検索して最初の結果をクリックしてください"）
4. "Start AI Agent" をクリック
5. AIがカーソルを動かして自動操作するのを観察

## 技術スタック 🔧

- **バックエンド**: Flask (Python)
- **ブラウザ制御**: Selenium + Chrome DevTools Protocol
- **AI モデル**: Llama 3.2 Vision 11B (Ollama)
- **画像処理**: Pillow
- **フロントエンド**: HTML/JavaScript (MJPEG streaming)

## 主な機能 🎨

### AIカーソル
- 80px巨大グロー効果
- 赤いクロスヘア
- スムーズな移動アニメーション

### 座標グリッド
- 黄色の100pxグリッド
- X/Y座標ラベル
- 中心マーカー

### ループ防止システム
1. 完全一致アクション検出
2. 近接座標検出（100px閾値）
3. 強制アクションオーバーライド

## API エンドポイント 📡

- `GET /` - メインUI
- `GET /stream` - MJPEGビデオストリーム
- `POST /ai/task` - AIタスク開始
- `GET /ai/status/<task_id>` - タスクステータス
- `POST /ai/stop/<task_id>` - タスク停止
- `GET /ai/health` - ヘルスチェック

## 設定 ⚙️

`app.py`で以下をカスタマイズ可能:

```python
OLLAMA_MODEL = "llama3.2-vision:11b"  # AIモデル
AI_MAX_STEPS = 30                     # 最大ステップ数
AI_STEP_TIMEOUT = 15                  # タイムアウト（秒）
JPEG_QUALITY = 65                     # ストリーム画質
TARGET_STREAM_WIDTH = 1024            # ストリーム幅
```

## ライセンス 📄

MIT License

## 注意事項 ⚠️

- これは開発サーバーです。本番環境では使用しないでください
- AIの操作は完璧ではありません
- ローカルでのみ動作します（セキュリティ上の理由）

## 貢献 🤝

プルリクエスト歓迎！

## 作者 👨‍💻

Created with ❤️ using AI assistance
