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
| M8 | TrackingService 3 モダリティ統合・TrackingFrame 公開 API | `main` (6a42336) |

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
  X・Y ともに `TextureConverter.ToTensor` が GPU→Tensor で変換する際に反転するため、`DecodeLandmarkResult` 内で `lx = 1f - raw_x / size`・`ly = 1f - raw_y / size` が**必要**（FaceLandmarker の ly と同構造、ただし lx も必要な点が異なる）。
  当初「Y 反転不要」と記録していたが、これは旧 `MakeRoi`（rotation≈-π）が cos(-π)=-1・sin(-π)=0 で X・Y を偶然打ち消していたための誤認。
- **Hand モデル（tf2onnx 変換 full 版）**：Unity HF モデルと同じ挙動（lx・ly ともに `1f -` 補正が必要）。
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
- presence に再度 `Sigmoid` をかけると低 confidence でも 0.5 付近になり、空/ズレ ROI の固定ランドマークを描き続けるため禁止。
- 正しい判定：`handednessRaw > 0.7f → Left（カメラ右）`、`< 0.3f → Right（カメラ左）`、それ以外 Unknown。

**NMS IoU 閾値**
- `k_NmsIouThresh = 0.3f` は厳しすぎ、両手が隣接すると 2 本目が suppressed される。
- `0.5f` に緩和すると 2 本目検出率が改善する。

**Hand ROI の回転・中心バグ（Unity BlazeHand sample 準拠）**
- `DecodeHandRoi` が BlazePose 用 `MakeRoi`（`-(Atan2(-dy,dx)) - π/2`）を流用していたため2つのバグがあった：
  1. **ROI 中心 = 手首**（正しくはボックス中心 = 手のひら中心）
  2. **回転式に -π/2 オフセット付き** → 手が真上を向くとき約 90° のズレが ProjectLandmark に生じる
- 左右非対称な症状（左手 OK、右手が 90° ずれ）は、左右で MCP の wrist からの dx 符号が逆になるため回転誤差量が異なることに起因。
- 修正：`BlazeUtils.MakeHandRoi` を追加。`boxSize = max(boxW, boxH)`、回転 = `π/2 - Atan2(dy, dx)`（手首 kp0→中指 MCP kp2 を ROI の Y 軸へ揃える）。
- ROI 中心は `boxCenter + 0.5 * boxSize * normalize(kp2 - kp0)` にずらしてから、`boxSize *= 2.6`。MediaPipe の `shift_y: -0.5` 相当で、指先を landmarker 入力に含めるため必須。

**Hand BlitRoi と ProjectLandmarkTopOrigin の整合性**
- Hand は `RotatedRoiBlit` shader で ROI クロップにも `roi.Rotation` を適用し、`ProjectLandmarkTopOrigin` も同じ `roi.Rotation` で逆射影する。
- shader が見つからない環境のみ軸沿いクロップへフォールバックし、その場合は `ProjectLandmarkTopOrigin` に `rotation=0` の ROI を渡す。
- クロップ側と逆射影側の rotation を片方だけ変えると、右手などで **90° ズレ**が再発する。

### M8 実装で確定した座標系まとめ

**Hand / Pose ランドマーク Y 軸：top-origin（0=上端）と確定**
- AlterEgo の `KpToVec` / `HkToVec` が `-kp.Y` を使用（3D 変換で符号反転）→ top-origin の証明。
- 正しい表示：OnGUI は `X * Width`, `Y * Height`（そのまま）、Hand GL は `GL.Vertex3(X, 1f - Y, 0f)`（Y のみ反転）。
- Face（tflite2onnx）は別物（bottom-origin、デコーダで `1f - raw/size`）。モデル出所によって慣習が異なる点に注意。

---

## 今後の計画

### M9：アバター適用
- `FaceFrame.Landmarks` → ARKit 52 Blendshape パラメータへの変換（表情推定）。
- `PoseFrame.Landmarks` → VRM / Humanoid ボーン回転への変換。
- `HandFrame.Landmarks` → 指 IK / ハンドシェイプ分類。

### 未検証・残タスク
- Profiler で GC Alloc / frame = 0 の実測確認（M7 以降未実施）。
- macOS での動作確認（`GPUCompute` → `GPUPixel` フォールバックの挙動含む）。
- ~~`HandTrackingDemo` の X 方向が正しく重なっているか実測確認。~~ → 確認済み（`1f - X` が必要）
- ~~Hand / Pose の Y 軸反転を解消する。~~ → 確認済み（OnGUI=`Y*H`、GL=`1f-Y`）
- `FaceTrackingDemo.unity` / `TrackingServiceDemo.unity` シーン（Inspector アセット割り当て）の設定手順を README に追記。
