// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Processor/PoseLandmarker.cs
//
// BlazePose（detector + landmarker）2 段の推論ラッパ。公開 API。
//
// 入力：RenderTexture（典型的には WebcamSource.Frame）
// 出力：33 点の <see cref="PoseKeypoint"/>（pre-allocated 配列を pull 型で返す）
//
// 設計方針：
// - GC Alloc を 0 にするため、Tensor / 出力配列は起動時 1 回だけ確保。
// - 推論完了は Awaitable<Tensor> で待つ（Sentis 純正 readback API）。
//   これにより async ValueTask スタイル（CLAUDE.md 規約）と整合し、コルーチンを使わない。
// - 検出器のスコア argmax は今フェーズでは CPU 側で実行（2254 floats なので小さい）。
//   後の最適化フェーズで Functional グラフ拡張に置き換える。
// - ROI クロップは M5 では軸沿い（rotation 0）の正方形クロップに簡略化。
//   landmarker 出力のローカル座標 → 入力画像座標への射影では rotation は適用するため、
//   検出器が示す傾きはランドマーク座標に反映される。
//   厳密な回転クロップは後続コミットでカスタムシェーダ Blit に置き換え予定。
//
// 必要な同梱物（Models/ に配置、ユーザーが手動 DL）：
//   - pose_detection.sentis        : detector 本体 (input 1×224×224×3)
//   - pose_landmarker_full.sentis  : landmarker 本体 (input 1×256×256×3)
//   - pose_anchors.csv             : 2254 行の anchor 定義
// 詳細は Models/README.md を参照。

using System;
using System.Threading;
using Moderato.AI.Tracking.Core;
using Unity.InferenceEngine;
using UnityEngine;

namespace Moderato.AI.Tracking.Processor
{
    /// <summary>
    /// BlazePose 2 段ランドマーカ。1 つの <see cref="PoseLandmarker"/> インスタンスは
    /// 1 つの Worker セット（detector + landmarker）と 1 セットの I/O テンソルを保有する。
    /// </summary>
    /// <remarks>
    /// スレッド安全ではない。Update スレッド（メインスレッド）からのみ呼ぶこと。
    /// Sentis の Worker / Tensor 自体がメインスレッド前提のため。
    /// </remarks>
    public sealed class PoseLandmarker : IDisposable
    {
        // ---- detector ------------------------------------------------------
        const int k_DetectorInputSize = 224;
        const int k_DetectorAnchorCount = 2254;
        // detector 出力ボックス 1 本のチャンネル数。
        // 形状は (1, 2254, 12) を想定：cx,cy,w,h, kp0x,kp0y,kp1x,kp1y,kp2x,kp2y,kp3x,kp3y
        const int k_DetectorBoxStride = 12;

        // ---- landmarker ----------------------------------------------------
        const int k_LandmarkerInputSize = 256;
        // (1, 195) = 33 keypoints × (x, y, z, visibility, presence) ではなく
        // BlazePose Full は (1, 195) = 39 × 5 だが、ユーザの目的は 33 点なので前 33 点のみ採用。
        // モデル種別により (1, 195) / (1, 165) のどちらも来うる。両方扱えるよう動的に解釈する。
        const int k_LandmarkerAttrCount = 5;

        readonly Worker m_DetectorWorker;
        readonly Worker m_LandmarkerWorker;

        // 起動時に確保し使い回す。
        readonly Tensor<float> m_DetectorInput;     // (1, 224, 224, 3) または model 期待形状
        readonly Tensor<float> m_LandmarkerInput;   // (1, 256, 256, 3) 同上
        readonly RenderTexture m_RoiTexture;        // 256×256 ROI 切り抜き用
        readonly PoseAnchor[] m_Anchors;
        readonly PoseKeypoint[] m_Landmarks;        // 33 点
        readonly TextureTransform m_DetectorTransform;
        readonly TextureTransform m_LandmarkerTransform;

        // ROI 計算過程の一時バッファ（ホットパスで new しないために field 化）
        float m_LastDetectionScore;
        bool m_HasValidDetection;

        bool m_Disposed;

        /// <summary>
        /// 各モデル・anchor を渡して構築する。
        /// </summary>
        /// <param name="detector">BlazePose detector の <see cref="ModelAsset"/>。Inspector で割当。</param>
        /// <param name="landmarker">BlazePose landmarker の <see cref="ModelAsset"/>。</param>
        /// <param name="anchorsCsv">anchors.csv の <see cref="TextAsset"/>（2254 行）。</param>
        /// <param name="backend">通常は <see cref="SentisBackendFactory.ResolveBest"/> の戻り値。</param>
        public PoseLandmarker(ModelAsset detector, ModelAsset landmarker, TextAsset anchorsCsv, BackendType backend)
        {
            if (detector == null) throw new ArgumentNullException(nameof(detector));
            if (landmarker == null) throw new ArgumentNullException(nameof(landmarker));
            if (anchorsCsv == null) throw new ArgumentNullException(nameof(anchorsCsv));

            // モデルロード（起動時 1 回のみ）。
            var detectorModel = ModelLoader.Load(detector);
            var landmarkerModel = ModelLoader.Load(landmarker);

            m_DetectorWorker = new Worker(detectorModel, backend);
            m_LandmarkerWorker = new Worker(landmarkerModel, backend);

            // 入力テンソルを 1 回確保。BlazePose は NHWC 配置の (1, H, W, 3)。
            m_DetectorInput = new Tensor<float>(new TensorShape(1, k_DetectorInputSize, k_DetectorInputSize, 3));
            m_LandmarkerInput = new Tensor<float>(new TensorShape(1, k_LandmarkerInputSize, k_LandmarkerInputSize, 3));

            // ROI 切り抜き先 RT（landmarker と同サイズ）。Compute 直入力で使う。
            m_RoiTexture = new RenderTexture(k_LandmarkerInputSize, k_LandmarkerInputSize, 0, RenderTextureFormat.ARGB32)
            {
                name = "Moderato.AI.Tracking.PoseRoi",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            m_RoiTexture.Create();

            // anchors はテキストから 1 度だけパース。
            m_Anchors = PoseAnchors.Load(anchorsCsv.text);
            if (m_Anchors.Length != k_DetectorAnchorCount)
            {
                Debug.LogWarning(
                    $"[Moderato.AI.Tracking] Anchor count mismatch: csv={m_Anchors.Length} expected={k_DetectorAnchorCount}. " +
                    "Detection may misalign — verify the anchors.csv corresponds to this detector model.");
            }

            // ランドマーク格納先を確保（毎フレーム再利用）。
            m_Landmarks = new PoseKeypoint[TrackingConstants.PoseLandmarkCount];

            // BlazePose は NHWC、座標原点は左上。
            // TextureTransform は struct なので field 化 = 起動時値型コピーのみ（heap alloc なし）。
            m_DetectorTransform = new TextureTransform()
                .SetDimensions(k_DetectorInputSize, k_DetectorInputSize, 3)
                .SetTensorLayout(0, 3, 1, 2); // N=0, C=3, H=1, W=2 → NHWC
            m_LandmarkerTransform = new TextureTransform()
                .SetDimensions(k_LandmarkerInputSize, k_LandmarkerInputSize, 3)
                .SetTensorLayout(0, 3, 1, 2);
        }

        /// <summary>
        /// 1 フレームの推論を実行する。CLAUDE.md の指針に従い <see cref="Awaitable{T}"/> ベース。
        /// </summary>
        /// <param name="frame">入力フレーム。WebcamSource.Frame をそのまま渡す想定。</param>
        /// <param name="cancellationToken">中断トークン（任意）。</param>
        /// <returns>33 点ランドマークを含む <see cref="PoseFrame"/>。配列は内部バッファの参照。</returns>
        public async Awaitable<PoseFrame> DetectAsync(RenderTexture frame, CancellationToken cancellationToken = default)
        {
            if (m_Disposed) throw new ObjectDisposedException(nameof(PoseLandmarker));
            if (frame == null) throw new ArgumentNullException(nameof(frame));

            cancellationToken.ThrowIfCancellationRequested();

            // ---- 1) 入力 RenderTexture → detector 入力 Tensor（GPU 直） ----
            TextureConverter.ToTensor(frame, m_DetectorInput, m_DetectorTransform);

            // ---- 2) detector を Schedule ----
            m_DetectorWorker.Schedule(m_DetectorInput);

            // detector の 0 番出力 = boxes (1, 2254, 12)、1 番出力 = scores (1, 2254, 1) を想定。
            // モデル仕様により index/順序が違う場合があるが、形状で同定するためまず両方 readback。
            using var boxesCpu = (Tensor<float>)await m_DetectorWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using var scoresCpu = (Tensor<float>)await m_DetectorWorker.PeekOutput(1).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            // ---- 3) CPU 側で argmax + ROI 計算 ----
            // ReadOnlySpan<T> は ref struct なので async メソッドのローカルにできない (CS4012)。
            // よって Span を扱う処理はすべて同期ヘルパに切り出し、await をまたがない区間に閉じる。
            bool valid = TryDecodeRoi(scoresCpu, boxesCpu, m_Anchors, k_DetectorInputSize,
                out float bestScore, out RotatedRect roi);
            m_LastDetectionScore = bestScore;

            // 信頼度が低ければ無効フレームとして返す。閾値は MediaPipe のデフォルト 0.5。
            if (!valid)
            {
                m_HasValidDetection = false;
                return new PoseFrame(m_Landmarks, bestScore, false);
            }

            // ---- 4) ROI を 256×256 RT にクロップ Blit ----
            //
            // 入力 frame は任意サイズ。detector への入力では 224×224 にスケール済み。
            // ここでは「入力 frame 上の正規化座標」での ROI を計算し、Blit の scale/offset で軸沿いクロップする。
            // 厳密な回転クロップは後の改善で対応（CLAUDE.md 規約：都度コミットで段階的に進める）。
            BlitRoi(frame, m_RoiTexture, in roi, k_DetectorInputSize);

            // ---- 5) ROI RT → landmarker 入力 Tensor ----
            TextureConverter.ToTensor(m_RoiTexture, m_LandmarkerInput, m_LandmarkerTransform);

            // ---- 6) landmarker を Schedule ----
            m_LandmarkerWorker.Schedule(m_LandmarkerInput);
            using var landmarksCpu = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            // ---- 7) 33 点を pre-allocated 配列に展開（同期ヘルパに委譲）----
            DecodeLandmarks(landmarksCpu, in roi, k_DetectorInputSize, k_LandmarkerInputSize, m_Landmarks);

            m_HasValidDetection = true;
            return new PoseFrame(m_Landmarks, bestScore, true);
        }

        // ----------------------------------------------------------------------
        // 同期ヘルパ群。ReadOnlySpan<T> を async メソッドのローカルにできない
        // (CS4012) ため、Span を触る処理はすべてここに集約する。
        // すべて static で副作用なし（出力は引数経由）。
        // ----------------------------------------------------------------------

        /// <summary>
        /// detector 出力テンソルから最良検出を選び、ROI まで導出する。
        /// </summary>
        /// <returns>有効な検出があれば true。false なら出力は default。</returns>
        static bool TryDecodeRoi(
            Tensor<float> scores,
            Tensor<float> boxes,
            PoseAnchor[] anchors,
            float inputSize,
            out float bestScore,
            out RotatedRect roi)
        {
            ReadOnlySpan<float> scoresSpan = scores.AsReadOnlySpan();

            // CPU 直接参照の Span を取り（DownloadToArray は GC alloc するので使わない）、
            // 線形ループで argmax。LINQ / foreach も boxing の可能性があるため避ける。
            int bestIdx = -1;
            float bestRawScore = float.NegativeInfinity;
            for (int i = 0; i < scoresSpan.Length; i++)
            {
                float v = scoresSpan[i];
                if (v > bestRawScore) { bestRawScore = v; bestIdx = i; }
            }
            bestScore = BlazeUtils.Sigmoid(bestRawScore);

            if (bestIdx < 0 || bestIdx >= anchors.Length || bestScore < 0.5f)
            {
                roi = default;
                return false;
            }

            // boxes も Span 直参照。Slice は struct 操作のみで heap alloc なし。
            ReadOnlySpan<float> boxesSpan = boxes.AsReadOnlySpan();
            ReadOnlySpan<float> rawSpan = boxesSpan.Slice(bestIdx * k_DetectorBoxStride, k_DetectorBoxStride);
            var anchor = anchors[bestIdx];

            BlazeUtils.DecodeBox(
                rawSpan,
                anchor.CenterX, anchor.CenterY,
                inputSize,
                out _, out _, out _, out _,
                out float midHipX, out float midHipY,
                out float fullBodyX, out float fullBodyY);

            roi = BlazeUtils.MakeRoi(midHipX, midHipY, fullBodyX, fullBodyY);
            return true;
        }

        /// <summary>
        /// 検出された ROI を src から dst に軸沿いでクロップする（Blit）。
        /// 回転は <see cref="DecodeLandmarks"/> 側で出力座標に適用する。
        /// </summary>
        static void BlitRoi(RenderTexture src, RenderTexture dst, in RotatedRect roi, float inputSize)
        {
            float roiSize = Mathf.Max(roi.Width, roi.Height);
            float roiHalfNorm = (roiSize * 0.5f) / inputSize;
            float roiCenterUNorm = roi.CenterX / inputSize;
            float roiCenterVNorm = 1f - (roi.CenterY / inputSize); // Blit の V 軸は下原点

            float u0 = Mathf.Clamp01(roiCenterUNorm - roiHalfNorm);
            float v0 = Mathf.Clamp01(roiCenterVNorm - roiHalfNorm);
            float uSize = Mathf.Min(roiHalfNorm * 2f, 1f - u0);
            float vSize = Mathf.Min(roiHalfNorm * 2f, 1f - v0);

            // Graphics.Blit(src, dst, scale, offset) — ホットパスで GC alloc を出さないオーバーロード。
            Graphics.Blit(src, dst,
                new Vector2(uSize, vSize),
                new Vector2(u0, v0));
        }

        /// <summary>
        /// landmarker 出力 (1, N) を 33 点 <see cref="PoseKeypoint"/> に展開し、
        /// ROI で逆射影して入力画像正規化座標に戻す。
        /// </summary>
        static void DecodeLandmarks(
            Tensor<float> landmarks,
            in RotatedRect roi,
            float detectorInputSize,
            float landmarkerInputSize,
            PoseKeypoint[] dst)
        {
            ReadOnlySpan<float> lmSpan = landmarks.AsReadOnlySpan();
            for (int i = 0; i < TrackingConstants.PoseLandmarkCount; i++)
            {
                int o = i * k_LandmarkerAttrCount;
                float lx = lmSpan[o + 0] / landmarkerInputSize;
                float ly = lmSpan[o + 1] / landmarkerInputSize;
                float lz = lmSpan[o + 2] / landmarkerInputSize;
                float visibility = BlazeUtils.Sigmoid(lmSpan[o + 3]);
                float presence = BlazeUtils.Sigmoid(lmSpan[o + 4]);

                // 入力画像座標へ射影（rotation はここで適用）。
                Vector2 projected = BlazeUtils.ProjectLandmark(lx, ly, in roi, detectorInputSize);

                dst[i] = new PoseKeypoint(projected.x, projected.y, lz, visibility, presence);
            }
        }

        /// <summary>直近 <see cref="DetectAsync"/> の検出スコア。Profiler / デバッグ用。</summary>
        public float LastDetectionScore => m_LastDetectionScore;

        /// <summary>直近検出が信頼度しきい値を超えたか。</summary>
        public bool HasValidDetection => m_HasValidDetection;

        public void Dispose()
        {
            if (m_Disposed) return;
            m_Disposed = true;

            m_DetectorWorker?.Dispose();
            m_LandmarkerWorker?.Dispose();
            m_DetectorInput?.Dispose();
            m_LandmarkerInput?.Dispose();

            if (m_RoiTexture != null)
            {
                m_RoiTexture.Release();
                UnityEngine.Object.Destroy(m_RoiTexture);
            }
        }
    }
}
