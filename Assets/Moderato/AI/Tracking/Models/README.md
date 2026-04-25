# Moderato.AI.Tracking — Models

このディレクトリには Sentis 形式（`.sentis`）に変換済みのモデル本体と、
detector が使う anchor 定義 CSV を配置する。**ファイル本体はリポジトリに含めず、
利用者が HuggingFace から手動でダウンロードして配置する**運用とする。

すべて Apache-2.0 で配布されている Unity 公式変換済みアセットを使用する。
ダウンロードしたら **Git LFS** で管理される（リポジトリルートの `.gitattributes` で
`*.sentis` / `*.onnx` / `*.tflite` / `*.pb` が LFS 対象に設定済み）。

## ポーズ（M5）

ソース：[`unity/sentis-blaze-pose`](https://huggingface.co/unity/sentis-blaze-pose)

このディレクトリ直下に以下を配置：

| ファイル名 | 内容 | 推定サイズ |
|---|---|---|
| `pose_detection.sentis` | BlazePose detector（入力 1×224×224×3、出力 boxes (1,2254,12) + scores (1,2254,1)） | 〜2 MB |
| `pose_landmarker_full.sentis` | BlazePose landmarker Full（入力 1×256×256×3、出力 (1, 195) ＝39 keypoints×5 のうち先頭 33 点を使用） | 〜10 MB |
| `pose_anchors.csv` | detector の 2254 anchor 定義（`cx,cy,w,h` 1 行 1 anchor） | 〜80 KB |

軽量化したい場合は `pose_landmarker_lite.sentis` に差し替え可能（Lite/Full は
PoseLandmarker 側で出力形状が一致していれば動作する）。

### 配置後の手順

1. 上記 3 ファイルを `Assets/Moderato/AI/Tracking/Models/` に置く
2. Unity Editor で Project ウィンドウをリフレッシュ（自動でインポートされる）
3. `pose_anchors.csv` は `TextAsset` として読み込まれる（拡張子 `.csv` を `.txt` にリネームする必要はない。
   Unity は `.csv` を `TextAsset` として扱う）
4. `PoseLandmarker` のコンストラクタ引数として 3 つを渡す：
   ```csharp
   var pose = new PoseLandmarker(
       detector:    poseDetectionAsset,
       landmarker:  poseLandmarkerAsset,
       anchorsCsv:  poseAnchorsCsv,
       backend:     SentisBackendFactory.ResolveBest());
   ```

## 手（M6 で追加予定）

ソース：[`unity/sentis-hand-landmark`](https://huggingface.co/unity/sentis-hand-landmark)

## 顔（M7 で追加予定）

ソース：[`unity/sentis-blaze-face`](https://huggingface.co/unity/sentis-blaze-face) と
[`unity/sentis-face-landmarker`](https://huggingface.co/unity/sentis-face-landmarker) の組み合わせ。

## ライセンス

Unity 公式 HF リポジトリの全モデルは Apache-2.0。本パッケージのコード（MIT）と
混在しても再配布上の問題はないが、利用者がアプリにバンドルする場合は
Apache-2.0 の表記義務（NOTICE）を満たすこと。
