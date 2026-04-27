# Moderato.AI.Tracking — Claude Code ガイド

## プロジェクトの目的

Web カメラ映像から人間の **ポーズ（BlazePose, 33 点）/ 手（Hand Landmark, 21 点 × 両手）/ 顔（FaceMesh / Blendshape）** をリアルタイムに推定し、Unity シーン内のアバター・演出に反映する。

## 実行環境

- Unity Editor：`6000.4.4f1`（`ProjectSettings/ProjectVersion.txt` を参照）
- 対象プラットフォーム：**Windows / macOS スタンドアロン**
- 推論：**Unity Sentis（`com.unity.ai.inference 2.6.1`）** / バックエンドは `BackendType.GPUCompute`
- Barracuda は使用しない（Sentis に一本化）

## コーディング方針（厳守）

1. **非同期は `ValueTask` を使う。`IEnumerator` / Coroutine は原則禁止。**
   - 理由：コルーチンのステートマシン & `yield` はホットパスで GC Alloc の温床になるため。
   - `async ValueTask` を用い、`AsyncValueTaskMethodBuilder` のゼロアロケーション特性を活かす。必要なら UniTask の導入も検討。
2. **GC Alloc をホットパスで出さない（0 alloc 目標）。**
   - `Tensor` / `ComputeBuffer` / ランドマーク配列は起動時に確保して使い回す。
   - `$"..."` 文字列補間・`List<T>` の動的増加・boxing を伴う `foreach` は `Update()` 経路に置かない。
   - 開発時は Profiler の `GC Alloc` 列を常時 0 に保つ。
3. **推論は非同期 readback 前提**：`Worker.Schedule()` → 非同期 readback を `ValueTask` で包む。`PeekOutput()` の同期 readback は禁止。
4. **GPU テクスチャ直入力**：`TextureConverter.ToTensor(RenderTexture, ...)` を使い、CPU 経由のコピーを作らない。

## ディレクトリ構成（[`Moderato.Mathematics.FFT`](https://github.com/SingUp009/Moderato.Mathematics.FFT) 規約に準拠）

- **パッケージルート：`Assets/Moderato/`**（`package.json` はここ）
- Git URL インポート：`?path=Assets/Moderato`
- **コアロジックは必ず `Assets/Moderato/AI/Tracking/` 配下に置く。** `Assets/Scenes/` や `Assets/Scripts/Tests/` などパッケージ外には依存しない。
- 階層はフラット：`Runtime/` / `Editor/` ラッパは作らない。カテゴリ直下に `.asmdef` を置く。
- 公開 API は `AI/Tracking/Processor/`、内部実装は `AI/Tracking/Core/`（`internal` で閉じる）。
- 同梱 `.sentis` モデルは `AI/Tracking/Models/`（Git LFS 管理）。
- 動作確認用 Scene / 実験コードは `Assets/Scenes/Sandbox/` などパッケージ外に置く。
- テストは `Assets/Scripts/Tests/` にパッケージ外で配置し、`UNITY_INCLUDE_TESTS` define で切り替え。

## バージョン管理

- **Git + Git LFS**（`.onnx` / `.sentis` / 画像 / 動画 / FBX / wav は LFS）
- `.unity` / `.prefab` / `.asset` の merge driver は **UnityYAMLMerge**
- **各作業単位で都度コミットする。** マイルストーン（モデル単体完成・パイプライン統合など）ごとに必ずコミットを残す。
- コミットメッセージ prefix：`feat:` / `fix:` / `chore:` / `refactor:` / `perf:` / `test:`

## パッケージ配布（Git URL インポート）

- `Assets/Moderato/` 全体を UPM パッケージとして配布：

  ```
  https://github.com/<owner>/Moderato.AI.Tracking.git?path=Assets/Moderato
  ```

- **パッケージ境界を厳守**：コアロジックは `Assets/Moderato/AI/Tracking/` の内側で完結させる。`using` / `Resources.Load` / `AssetDatabase.FindAssets` などあらゆる依存経路でパッケージ外（`Assets/Scripts/`, `Assets/Scenes/`, `Assets/Resources/`）を参照しないこと。Git URL 経由で消費側が動かなくなる。

## 検収条件

新機能を `main` にマージする前に、以下を満たすこと：

- Profiler の `GC Alloc / frame` がゼロ
- 目標 30 fps が Windows / macOS で維持できている
- 3 モダリティ（ポーズ・手・顔）が同時に動作している

---

## 実装進捗

### 完成済みマイルストーン

| M# | 内容 | ブランチ / コミット |
|---|---|---|
| M5 | BlazePose 33 点ポーズ推定 | `main` (f0a10da 以前) |
| M6 | BlazeHand 21 点 × 両手（NMS）推定 | `main` (9b6fe21) |
| M7 | BlazeFace + FaceMesh 468 点顔推定 | `main` (0710b70〜) |

### M7 実装で判明した座標系の罠（次回参照用）

**モデル変換**
- `tensorflow` は Python 3.12+ では入らない。`tflite2onnx`（TF 不要）で代替可能：
  ```
  pip install tflite2onnx onnx
  python -c "import tflite2onnx; tflite2onnx.convert('face_landmark.tflite', 'face_landmark.onnx')"
  ```
- `tflite2onnx` は TFLite の **Y=0=画像上端（標準画像座標）** をそのまま保持する。
- Unity 公式 HuggingFace モデル（hand / pose）は Unity の **Y=0=テクスチャ下端** 慣習に合わせて変換済みのため Y 反転不要。

**tflite2onnx 変換モデルの Y 反転**
- `DecodeLandmarks` 内で `ly = 1f - raw_y / landmarkerInputSize` が必要。
- `ProjectLandmark` は `roi.CenterY`（Y=0=top ピクセル空間）を使うが、反転後の `ly` を渡しても
  `OnGUI` 側の `(1f - lm.Y) * Screen.height` と合わせて結果的に正しく表示される。

**WebCamTexture の X 方向とモデルごとの差異**
- `WebCamTexture` の表示は（前面カメラでは）**ミラー表示**になる。
- **Hand モデル（Unity HuggingFace）**：ランドマーク X は非ミラー空間で出力される。
  → `HandTrackingDemo` の OnGUI / GL 描画で `1f - X` の反転が**必要**。
- **Hand モデル（tf2onnx 変換 full 版）**：Y 反転不要。X 反転（`1f - X`）が必要。Unity HF モデルと同じ挙動。
- **Face モデル（tflite2onnx 変換）**：ランドマーク X はミラー表示と同じ空間で出力される。
  → `FaceTrackingDemo` の描画で `1f - X` の反転は**不要**（`lm.X * Screen.width` のみ）。
- モデルの出所（Unity 公式変換 / tflite2onnx / tf2onnx）によって X・Y 慣習が異なる。追加モデルを組み込む際は実測で確認すること。

**MakeFaceRoi の回転符号**
- 正しい式（MediaPipe 準拠）：`rotation = Mathf.Atan2(-dy, dx)`
- `-(Mathf.Atan2(-dy, dx))` と書くと符号が逆になり、頭を傾けるとランドマークが逆方向に傾く。
- `MakeRoi`（BlazePose 用）は `-π/2` オフセットあり・符号は同様なので別物として扱うこと。

**検出閾値**
- `k_DetectThresh = 0.5f` が MediaPipe デフォルト。`0.75f` は高すぎて頻繁に検出が途切れる。

**Hand モデルのモデル同一性（M6→"Full" 切り替え検証）**
- `palm_detection_full.onnx`（tf2onnx 変換）と `hand_detector.onnx`（Unity HF lite）は**実質同一**（ファイルサイズ差 4 バイトのみ、同一入力で完全に同じ出力）。
- `hand_landmark_full.onnx` と `hand_landmarks_detector.onnx` も**実質同一**（差 7 バイト）。
- 「full に切り替えたら挙動が変わった」場合、モデル側の問題ではなくコード側のバグを疑うこと。

**handedness 出力の型（post-sigmoid [0,1]）**
- BlazeHand の handedness / presence 出力は **sigmoid 適用済みの [0,1] 値**で出力される。
- raw logit と誤解して `< -0.5f` で Right 判定すると**永遠に true にならず、右手が常に Unknown** になるバグが生じる。
- 正しい判定：`handednessRaw > 0.7f → Left（カメラ右）`、`< 0.3f → Right（カメラ左）`、それ以外 Unknown。

**NMS IoU 閾値**
- `k_NmsIouThresh = 0.3f` は厳しすぎ、両手が隣接すると 2 本目が suppressed される。
- `0.5f` に緩和すると 2 本目検出率が改善する。

**Hand ROI の回転バグ（MakeRoi vs MakeHandRoi）**
- `DecodeHandRoi` が BlazePose 用 `MakeRoi`（`-(Atan2(-dy,dx)) - π/2`）を流用していたため2つのバグがあった：
  1. **ROI 中心 = 手首**（正しくはボックス中心 = 手のひら中心）
  2. **回転式に -π/2 オフセット付き** → 手が真上を向くとき約 90° のズレが ProjectLandmark に生じる
- 左右非対称な症状（左手 OK、右手が 90° ずれ）は、左右でインデックス MCP の wrist からの dx 符号が逆になるため回転誤差量が異なることに起因。
- 修正：`BlazeUtils.MakeHandRoi` を追加（`MakeFaceRoi` と同構造）。ROI 中心 = ボックス中心、回転 = `Atan2(-dy, dx)`（オフセットなし）。`DecodeHandRoi` はこれを使う。

**BlitRoi と ProjectLandmark の整合性（rotation=0 必須）**
- `BlitRoi` は **軸沿い（回転なし）クロップ**。`roi.Rotation` を Blit に適用していない。
- `ProjectLandmark` に非ゼロの `roi.Rotation`（例：手が直立なら≈π/2）を渡すと、クロップ座標の Y オフセットが回転行列でX方向に混入し **90° ズレ**が生じる（face では回転≈0 なので誤差が小さく表面化しにくい）。
- 修正：`DecodeLandmarkResult` 内で `ProjectLandmark` に渡す ROI は `rotation=0` に固定した `axisRoi` を使う（実際のクロップが回転していないため）。
- `MakeHandRoi` が計算する回転情報は将来の回転 Blit 実装のために保持する。

---

## 今後の計画

### M8：3 モダリティ統合（TrackingService）
- `PoseLandmarker` / `HandLandmarker` / `FaceLandmarker` を 1 つの `TrackingService` に束ねる。
- 各 Processor を並列スケジュール（`Worker.Schedule` は GPU 非同期なのでフレーム内に複数発行可能）。
- 公開 API 案：`TrackingService.DetectAsync(RenderTexture) → TrackingFrame`

### M9：アバター適用
- `FaceFrame.Landmarks` → ARKit 52 Blendshape パラメータへの変換（表情推定）。
- `PoseFrame.Landmarks` → VRM / Humanoid ボーン回転への変換。
- `HandFrame.Landmarks` → 指 IK / ハンドシェイプ分類。

### 未検証・残タスク
- Profiler で GC Alloc / frame = 0 の実測確認（M7 以降未実施）。
- macOS での動作確認（`GPUCompute` → `GPUPixel` フォールバックの挙動含む）。
- ~~`HandTrackingDemo` の X 方向が正しく重なっているか実測確認。~~ → 確認済み（`1f - X` が必要）
- `FaceTrackingDemo.unity` シーン（Inspector アセット割り当て）の設定手順を README に追記。
