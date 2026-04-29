# Moderato.AI.Tracking — エージェント作業ガイド

このドキュメントは AI エージェントがこのリポジトリで作業するための実務リファレンスです。
プロジェクト全体の方針は `CLAUDE.md` を参照してください。

---

## プロジェクト 30 秒サマリ

Unity 6 上で WebCam 映像から **ポーズ（33pt）/ 手（21pt×2）/ 顔（468pt）** をリアルタイム推定する。
推論エンジンは **Unity Sentis 2.6.1**（ONNX モデル、GPUCompute バックエンド）。
M8 時点で 3 モダリティの統合 API（`TrackingService`）が完成。表示の Y 軸反転が未解決。

---

## ファイルマップ

```
Assets/Moderato/AI/Tracking/
├── Moderato.AI.Tracking.cs        公開型定義（全 Frame/Keypoint struct、TrackingFrame）
├── Moderato.AI.Tracking.asmdef    アセンブリ定義
├── Core/
│   ├── BlazeUtils.cs              Sigmoid / DecodeBox / MakeRoi / ProjectLandmark など
│   ├── PoseAnchors.cs             anchor CSV ローダ
│   ├── SentisBackendFactory.cs    GPUCompute → GPUPixel → CPU のフォールバック選択
│   └── WebcamSource.cs            WebCamTexture → RenderTexture ポンプ
├── Processor/
│   ├── PoseLandmarker.cs          BlazePose 2 段（detector 224² + landmarker 256²）
│   ├── HandLandmarker.cs          BlazeHand 2 段 + NMS 2 手（detector 192² + landmarker 224²）
│   ├── FaceLandmarker.cs          BlazeFace 2 段（detector 128² + landmarker 192²）
│   └── TrackingService.cs         3 Processor を束ねる統合 API（M8）
└── Models/
    ├── pose_detection.onnx
    ├── pose_landmarks_detector_lite.onnx
    ├── anchors.csv
    ├── hand_detector.onnx
    ├── hand_landmarks_detector.onnx
    ├── palm_detection_anchors.csv
    ├── blaze_face_short_range.onnx
    ├── face_landmark.onnx
    └── face_detection_anchors.csv

Assets/Scenes/Sandbox/
    ├── HandTrackingDemo.cs/.unity
    ├── FaceTrackingDemo.cs/.unity
    └── TrackingServiceDemo.cs
        TrackingServiceDemo/TrackingServiceDemo.unity
```

---

## 公開 API 早見表

```csharp
// 統合（M8）
var service = new TrackingService(
    poseDetector, poseLandmarker, poseAnchors,
    palmDetector, handLandmarker, palmAnchors,
    faceDetector, faceLandmarker, faceAnchors,
    BackendType.GPUCompute);

TrackingFrame frame = await service.DetectAsync(renderTexture, ct);
frame.Pose.Landmarks[i]      // PoseKeypoint { X, Y, Z, Visibility, Presence }
frame.Hands[h].Landmarks[i]  // HandKeypoint { X, Y, Z }  h=0/1
frame.Face.Landmarks[i]      // FaceKeypoint { X, Y, Z }

// 個別
PoseFrame   pose  = await poseLandmarker.DetectAsync(rt, ct);
HandFrame[] hands = await handLandmarker.DetectAsync(rt, ct);
FaceFrame   face  = await faceLandmarker.DetectAsync(rt, ct);
```

- 全 Landmark X・Y は入力画像正規化座標 `[0, 1]`。
- `Hands` は常に長さ 2 の配列（内部 pre-alloc）。`IsValid` で有効性を確認。
- **戻り値の配列は内部バッファの参照**。フレームをまたいでキャッシュしないこと。

---

## 座標系リファレンス（最重要・バグの温床）

### 確定済み

| 項目 | 値 | 根拠 |
|---|---|---|
| Face landmark Y | **bottom-origin**（0=下端、1=上端） | `ly = 1f - raw_y / size` で変換、`(1f - Y) * Height` で正常表示確認済み |
| Face landmark X | ミラー空間（カメラ左=小、右=大） | `FaceTrackingDemo` で X 補正なしで正常と確認 |
| Hand landmark X | 非ミラー空間 | `HandTrackingDemo` GL で `1f - X` 補正が必要と確認済み |
| Face OnGUI 描画 | `px = X * W`、`py = (1f - Y) * H` | 正常表示確認済み |
| Face GL 描画 | `GL.Vertex3(X, Y, 0f)` | X 補正なし、Y は GL bottom-origin と一致 |
| Hand landmark Y | **top-origin**（0=上端、1=下端） | AlterEgo の `HkToVec` が `-kp.Y` を使用していることで確認（bottom-origin なら `+kp.Y`） |
| Pose landmark Y | **top-origin**（0=上端、1=下端） | AlterEgo の `KpToVec` が `-kp.Y` を使用していることで確認 |
| Hand OnGUI 描画 | `px = X * W`、`py = Y * H` | top-origin で OnGUI と一致するためそのまま |
| Pose OnGUI 描画 | `px = X * W`、`py = Y * H`（visibility >= 0.5 のみ） | 同上 |
| Hand GL 描画 | `GL.Vertex3(1f - X, 1f - Y, 0f)` | X ミラー補正 + Y を 1f- で top→bottom 変換 |
| Pose GL 描画 | `GL.Vertex3(1f - X, 1f - Y, 0f)` | 同上 |

---

## 推論パイプラインのパターン

各 Processor は以下の 2 段構成で統一されている：

```
RenderTexture
  → TextureConverter.ToTensor()               # GPU 直入力（CPU コピー禁止）
  → Worker.Schedule()                         # GPU 非同期スケジュール
  → await PeekOutput().ReadbackAndCloneAsync() # 非同期 readback
  → CPU 上で ROI 計算（BlazeUtils.Make*Roi）
  → BlitRoi()                                 # 軸沿いクロップ（rotation=0 固定）
  → TextureConverter.ToTensor()
  → Worker.Schedule()
  → await ReadbackAndCloneAsync()
  → DecodeLandmarks()                         # ReadOnlySpan → pre-alloc 配列
```

**TrackingService の GPU 並列化**:
```csharp
// 各 DetectAsync は最初の await（Detector readback）まで同期実行される。
// 3 行の呼び出しでフレーム内に全 Detector GPU ジョブが先行投入される。
var pa = poseLandmarker.DetectAsync(frame, ct);
var ha = handLandmarker.DetectAsync(frame, ct);
var fa = faceLandmarker.DetectAsync(frame, ct);
var pose  = await pa;
var hands = await ha;
var face  = await fa;
```

---

## コーディング必須ルール（CLAUDE.md より抜粋）

1. **`async Awaitable<T>` を使う**。`IEnumerator` / Coroutine 禁止。
2. **ホットパスで GC Alloc ゼロ**。Tensor / 配列はコンストラクタで確保し使い回す。
3. **`PeekOutput()` の同期 readback 禁止**。必ず `ReadbackAndCloneAsync()` を使う。
4. **`ReadOnlySpan` は `async` メソッドのローカルに置けない**（CS4012）。Span を触る処理は静的同期ヘルパに切り出す。
5. **パッケージ境界を厳守**：`Assets/Moderato/` 内から `Assets/Scenes/` 等を参照しない。

---

## 実装状態

| # | 内容 | 状態 |
|---|---|---|
| M5 | BlazePose 33 点ポーズ推定 | ✅ 完了 |
| M6 | BlazeHand 21 点 × 両手（NMS） | ✅ 完了 |
| M7 | BlazeFace + FaceMesh 468 点 | ✅ 完了 |
| M8 | TrackingService 統合・TrackingFrame 公開 API | ✅ 完了 |
| — | Hand / Pose Y 軸表示反転の修正 | ✅ 完了（候補 B: OnGUI=`Y*H`、GL=`1f-Y`） |
| M9 | アバター適用（Blendshape / VRM ボーン / 指 IK） | 📋 未着手 |

---

## 次のアクション候補

### 優先度 高：M9 アバター適用
- `FaceFrame.Landmarks` → ARKit 52 Blendshape（Jaw, Eye, Brow など）
- `PoseFrame.Landmarks` → VRM Humanoid ボーン回転（T-pose 基準の相対回転）
- `HandFrame.Landmarks` → 指 IK / FingerPose 分類

### 優先度 低：品質
- Profiler GC Alloc / frame = 0 の実測確認（M7 以降未実施）
- macOS GPUPixel フォールバック確認

---

## デバッグ Tips

**コンパイルエラーの確認**（UnityMCP 使用時）
```
read_console(types=["error"])
```

**座標系の実測**
```csharp
// OnGUI の適当な場所に追加してログ表示
GUI.Label(new Rect(10, 60, 400, 22), $"hand[0] wrist Y = {hands[0].Landmarks[0].Y:F3}");
```
手を上に動かして Y 増加 → bottom-origin / 減少 → top-origin。

**モデル出力インデックスの確認**
```python
import onnx
m = onnx.load('model.onnx')
[print(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]
```
