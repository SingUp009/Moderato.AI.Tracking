# Moderato.AI.Tracking — Models

このディレクトリには推論モデル本体と、detector が使う anchor 定義 CSV を配置する。

## 方針

- **モデル本体（`.onnx` / `.sentis` / `.tflite` / `.pb`）はリポジトリに含めない。**
  リポジトリルートの `.gitignore` で `Assets/Moderato/AI/Tracking/Models/*.{onnx,sentis,tflite,pb}` と
  対応する `.meta` を git 管理対象から除外している。
  - 理由：合計サイズが数百 MB に達し、Apache-2.0 で配布されているソース（HuggingFace）への
    URL 参照で十分なため。LFS でも consumer 側の Git URL UPM import で大量 DL を強制する
    必要がない。
- **`anchors.csv` は git 管理する。** 80 KB 程度の小さなテキストデータで、コードが
  動作するために必須の anchor 定義（detector の出力チャンネル位置）であるため。
- 利用者は下記手順で本体を **手動 DL** して配置する。

すべて Sentis（`Unity.InferenceEngine`）の `ModelAsset` として読み込めるため、
`.onnx` のままで問題なく動作する（必要に応じて `.sentis` に変換してキャッシュも可）。

## ポーズ（M5）

ソース：[`unity/sentis-blaze-pose`](https://huggingface.co/unity/sentis-blaze-pose)
（Apache-2.0、Unity 公式変換済みアセット）

このディレクトリ直下に以下を配置：

| ファイル名 | 内容 | 推定サイズ | git 管理 |
|---|---|---|---|
| `pose_detection.onnx` | BlazePose detector（入力 1×224×224×3、出力 boxes (1,2254,12) + scores (1,2254,1)） | 〜15 MB | ❌ 除外 |
| `pose_landmarks_detector_lite.onnx` | landmarker Lite（軽量・速度優先） | 〜5 MB | ❌ 除外 |
| `pose_landmarks_detector_full.onnx` | landmarker Full（バランス） | 〜13 MB | ❌ 除外 |
| `pose_landmarks_detector_heavy.onnx` | landmarker Heavy（精度優先） | 〜53 MB | ❌ 除外 |
| `anchors.csv` | detector の 2254 anchor 定義（`cx,cy,w,h` 1 行 1 anchor） | 〜80 KB | ✅ 含める |

`pose_landmarks_detector_*.onnx` は 3 種類のうち 1 つを選んで使えばよい。
推奨は最初は `lite`、安定したら `full` に切り替え。`heavy` は単独 60fps を諦めるなら可。

### 配置後の手順

1. 上記から必要な `.onnx` を `Assets/Moderato/AI/Tracking/Models/` に置く
2. Unity Editor で Project ウィンドウをリフレッシュ（自動でインポートされる）
3. `anchors.csv` は `TextAsset` として読み込まれる
4. `PoseLandmarker` のコンストラクタ引数として渡す：
   ```csharp
   var pose = new PoseLandmarker(
       detector:    poseDetectionAsset,    // pose_detection.onnx
       landmarker:  poseLandmarkerAsset,   // pose_landmarks_detector_*.onnx
       anchorsCsv:  anchorsCsv,            // anchors.csv
       backend:     SentisBackendFactory.ResolveBest());
   ```

## 手（M6）

ソース：[`unity/sentis-hand-landmark`](https://huggingface.co/unity/sentis-hand-landmark)
（Apache-2.0、Unity 公式変換済みアセット）

このディレクトリ直下に以下を配置：

| ファイル名 | 内容 | git 管理 |
|---|---|---|
| `hand_detector.onnx` | BlazeHand palm detector（入力 1×192×192×3、出力 boxes (1,2016,18) + scores (1,2016,1)） | ❌ 除外 |
| `hand_landmarks_detector.onnx` | Hand landmark（入力 1×224×224×3、出力 landmarks (1,63) + handedness (1,1) + presence (1,1)） | ❌ 除外 |
| `palm_detection_anchors.csv` | palm detector の anchor 定義（`cx,cy,w,h`、2016 行） | ✅ 含める |

> **注意：** `pose_detection.onnx` 用の `anchors.csv`（2254 行）と
> `palm_detection_anchors.csv`（~2016 行）は**別ファイル**。混在させないこと。

### 配置後の手順

1. 上記の `.onnx` と `palm_detection_anchors.csv` を `Assets/Moderato/AI/Tracking/Models/` に置く
2. Unity Editor で Project ウィンドウをリフレッシュ
3. `HandLandmarker` のコンストラクタ引数として渡す：
   ```csharp
   var hand = new HandLandmarker(
       palmDetector:    palmDetectionAsset,    // hand_detector.onnx
       handLandmarker:  handLandmarkAsset,     // hand_landmarks_detector.onnx
       palmAnchorsCsv:  palmAnchorsCsvAsset,   // palm_detection_anchors.csv
       backend:         SentisBackendFactory.ResolveBest());
   ```
4. 毎フレーム `DetectAsync(webcam.Frame)` を呼ぶ。戻り値は `HandFrame[2]`（内部バッファ参照）。

## 顔（M7 で追加予定）

ソース：[`unity/sentis-blaze-face`](https://huggingface.co/unity/sentis-blaze-face) と
[`unity/sentis-face-landmarker`](https://huggingface.co/unity/sentis-face-landmarker) の組み合わせ。

## ライセンス

Unity 公式 HF リポジトリの全モデルは Apache-2.0。本パッケージのコード（MIT）と
混在しても再配布上の問題はないが、利用者がアプリにバンドルする場合は
Apache-2.0 の表記義務（NOTICE）を満たすこと。
