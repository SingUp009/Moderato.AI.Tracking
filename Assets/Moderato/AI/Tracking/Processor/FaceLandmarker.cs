// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Processor/FaceLandmarker.cs
//
// BlazeFace（detector + face landmark）2 段の推論ラッパ。公開 API。
// 単一顔検出。M7 v1。
//
// 必要な同梱物（Models/ に配置）：
//   - blaze_face_short_range.onnx    : face detector (input 1×128×128×3)
//   - face_landmark.onnx             : 468 点 face landmark (input 1×192×192×3)
//                                      MediaPipe face_landmark.tflite を tf2onnx で変換
//   - face_detection_anchors.csv     : 896 行の anchor 定義
//
// ★ PeekOutput インデックスについて ★
// tf2onnx 変換後のテンソル名・順序は変換環境により異なる。
// 下記コマンドで確認し、必要なら 0/1 を入れ替えること：
//   python -c "import onnx; m=onnx.load('face_landmark.onnx');
//     [print(o.name,[d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]"
// 期待値：出力 0 = landmarks (1, 1404)、出力 1 = face_flag (1, 1)

using System;
using System.Threading;
using Moderato.AI.Tracking.Core;
using Unity.InferenceEngine;
using UnityEngine;

namespace Moderato.AI.Tracking.Processor
{
    /// <summary>
    /// BlazeFace 2 段ランドマーカ。単一顔を検出する。
    /// </summary>
    /// <remarks>
    /// スレッド安全ではない。メインスレッドからのみ呼ぶこと。
    /// <see cref="DetectAsync"/> の戻り値 <see cref="FaceFrame.Landmarks"/> は内部バッファの参照。
    /// フレームをまたいでキャッシュしないこと。
    /// </remarks>
    public sealed class FaceLandmarker : IDisposable
    {
        // ---- face detector (BlazeFace short range) --------------------------
        const int   k_DetectorInputSize   = 128;
        const int   k_DetectorAnchorCount = 896;
        // 4(box cx,cy,w,h) + 6 kp × 2(x,y) = 16
        const int   k_DetectorBoxStride   = 16;

        // ---- face landmark --------------------------------------------------
        const int   k_LandmarkerInputSize = 192;
        const int   k_LandmarkCount       = TrackingConstants.FaceLandmarkCount; // 468
        const int   k_LandmarkAttrCount   = 3;   // x, y, z のみ

        // ---- detection parameters -------------------------------------------
        const float k_RoiScale       = 1.5f;
        const float k_DetectThresh   = 0.75f;
        const float k_PresenceThresh = 0.5f;

        readonly Worker m_DetectorWorker;
        readonly Worker m_LandmarkerWorker;

        readonly Tensor<float> m_DetectorInput;     // (1, 128, 128, 3)
        readonly Tensor<float> m_LandmarkerInput;   // (1, 192, 192, 3)
        readonly RenderTexture m_RoiTexture;        // 192×192 ROI

        readonly PoseAnchor[]  m_Anchors;
        readonly FaceKeypoint[] m_Landmarks;        // [468] 起動時確保・使い回し

        readonly TextureTransform m_DetectorTransform;
        readonly TextureTransform m_LandmarkerTransform;

        float m_LastDetectionScore;
        bool  m_HasValidDetection;
        bool  m_Disposed;

        /// <summary>
        /// 各モデル・anchor を渡して構築する。
        /// </summary>
        public FaceLandmarker(
            ModelAsset  faceDetector,
            ModelAsset  faceLandmarker,
            TextAsset   faceAnchorsCsv,
            BackendType backend)
        {
            if (faceDetector   == null) throw new ArgumentNullException(nameof(faceDetector));
            if (faceLandmarker == null) throw new ArgumentNullException(nameof(faceLandmarker));
            if (faceAnchorsCsv == null) throw new ArgumentNullException(nameof(faceAnchorsCsv));

            m_DetectorWorker   = new Worker(ModelLoader.Load(faceDetector),   backend);
            m_LandmarkerWorker = new Worker(ModelLoader.Load(faceLandmarker), backend);

            m_DetectorInput   = new Tensor<float>(new TensorShape(1, k_DetectorInputSize,   k_DetectorInputSize,   3));
            m_LandmarkerInput = new Tensor<float>(new TensorShape(1, k_LandmarkerInputSize, k_LandmarkerInputSize, 3));

            m_RoiTexture = new RenderTexture(k_LandmarkerInputSize, k_LandmarkerInputSize, 0, RenderTextureFormat.ARGB32)
            {
                name       = "Moderato.AI.Tracking.FaceRoi",
                wrapMode   = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            m_RoiTexture.Create();

            m_Anchors = PoseAnchors.Load(faceAnchorsCsv.text);
            if (m_Anchors.Length != k_DetectorAnchorCount)
            {
                Debug.LogWarning(
                    $"[Moderato.AI.Tracking] Face anchor count={m_Anchors.Length}, expected {k_DetectorAnchorCount}. " +
                    "Verify face_detection_anchors.csv matches the face detector model.");
            }

            m_Landmarks = new FaceKeypoint[k_LandmarkCount];

            m_DetectorTransform = new TextureTransform()
                .SetDimensions(k_DetectorInputSize, k_DetectorInputSize, 3)
                .SetTensorLayout(0, 3, 1, 2);
            m_LandmarkerTransform = new TextureTransform()
                .SetDimensions(k_LandmarkerInputSize, k_LandmarkerInputSize, 3)
                .SetTensorLayout(0, 3, 1, 2);
        }

        /// <summary>
        /// 1 フレームの推論を実行する。
        /// </summary>
        /// <param name="frame">入力フレーム（WebcamSource.Frame をそのまま渡す想定）。</param>
        /// <param name="cancellationToken">中断トークン（任意）。</param>
        /// <returns>468 点ランドマークを含む <see cref="FaceFrame"/>。配列は内部バッファの参照。</returns>
        public async Awaitable<FaceFrame> DetectAsync(
            RenderTexture frame,
            CancellationToken cancellationToken = default)
        {
            if (m_Disposed) throw new ObjectDisposedException(nameof(FaceLandmarker));
            if (frame == null) throw new ArgumentNullException(nameof(frame));

            cancellationToken.ThrowIfCancellationRequested();

            // ---- 1) RT → detector Tensor（GPU 直） ----
            TextureConverter.ToTensor(frame, m_DetectorInput, m_DetectorTransform);

            // ---- 2) face detector Schedule ----
            m_DetectorWorker.Schedule(m_DetectorInput);

            // ---- 3) 非同期 readback ----
            // インデックス 0 = boxes (1, 896, 16)、1 = scores (1, 896, 1)。
            // 変換結果が異なる場合は上部コメントの確認手順を参照。
            using var boxesCpu  = (Tensor<float>)await m_DetectorWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using var scoresCpu = (Tensor<float>)await m_DetectorWorker.PeekOutput(1).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            // ---- 4) CPU 側で ROI 計算 ----
            bool valid = TryDecodeRoi(scoresCpu, boxesCpu, m_Anchors, k_DetectorInputSize,
                out float bestScore, out RotatedRect roi);
            m_LastDetectionScore = bestScore;

            if (!valid)
            {
                m_HasValidDetection = false;
                return new FaceFrame(m_Landmarks, bestScore, 0f, false);
            }

            // ---- 5) ROI → 192×192 RT にクロップ ----
            BlitRoi(frame, m_RoiTexture, in roi, k_DetectorInputSize);

            // ---- 6) RT → landmarker Tensor ----
            TextureConverter.ToTensor(m_RoiTexture, m_LandmarkerInput, m_LandmarkerTransform);

            // ---- 7) landmarker Schedule ----
            m_LandmarkerWorker.Schedule(m_LandmarkerInput);

            // インデックス 0 = landmarks (1, 1404)、1 = face_flag (1, 1)。
            using var lmCpu   = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using var presCpu = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(1).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            // ---- 8) ランドマークを pre-allocated 配列に展開 ----
            DecodeLandmarks(lmCpu, presCpu, in roi,
                k_DetectorInputSize, k_LandmarkerInputSize,
                m_Landmarks, out float presenceScore, out bool isValid);

            m_HasValidDetection = isValid;
            return new FaceFrame(m_Landmarks, bestScore, presenceScore, isValid);
        }

        // -----------------------------------------------------------------------
        // 同期ヘルパ（ReadOnlySpan は async メソッドのローカルに置けない / CS4012）
        // -----------------------------------------------------------------------

        static bool TryDecodeRoi(
            Tensor<float> scores,
            Tensor<float> boxes,
            PoseAnchor[]  anchors,
            float         inputSize,
            out float     bestScore,
            out RotatedRect roi)
        {
            ReadOnlySpan<float> scoresSpan = scores.AsReadOnlySpan();

            int   bestIdx = -1;
            float bestRaw = float.NegativeInfinity;
            for (int i = 0; i < scoresSpan.Length; i++)
            {
                float v = scoresSpan[i];
                if (v > bestRaw) { bestRaw = v; bestIdx = i; }
            }
            bestScore = BlazeUtils.Sigmoid(bestRaw);

            if (bestIdx < 0 || bestIdx >= anchors.Length || bestScore < k_DetectThresh)
            {
                roi = default;
                return false;
            }

            ReadOnlySpan<float> boxesSpan = boxes.AsReadOnlySpan();
            ReadOnlySpan<float> rawSpan   = boxesSpan.Slice(bestIdx * k_DetectorBoxStride, k_DetectorBoxStride);
            var anchor = anchors[bestIdx];

            // DecodeBox は raw[0..7] を読む：cx,cy,w,h, kp0x,kp0y,kp1x,kp1y
            // BlazeFace kp0 = right eye、kp1 = left eye（ストライド 16 の先頭 8 float を利用）
            BlazeUtils.DecodeBox(
                rawSpan,
                anchor.CenterX, anchor.CenterY,
                inputSize,
                out float boxCx,      out float boxCy,
                out float boxW,       out float boxH,
                out float rightEyeX,  out float rightEyeY,
                out float leftEyeX,   out float leftEyeY);

            roi = BlazeUtils.MakeFaceRoi(
                boxCx, boxCy, boxW, boxH,
                rightEyeX, rightEyeY, leftEyeX, leftEyeY,
                k_RoiScale);
            return true;
        }

        static void DecodeLandmarks(
            Tensor<float>  landmarks,
            Tensor<float>  presence,
            in RotatedRect roi,
            float          detectorInputSize,
            float          landmarkerInputSize,
            FaceKeypoint[] dst,
            out float      presenceScore,
            out bool       isValid)
        {
            presenceScore = BlazeUtils.Sigmoid(presence.AsReadOnlySpan()[0]);
            isValid = presenceScore >= k_PresenceThresh;

            ReadOnlySpan<float> lmSpan = landmarks.AsReadOnlySpan();
            for (int i = 0; i < k_LandmarkCount; i++)
            {
                int   o  = i * k_LandmarkAttrCount;
                float lx = lmSpan[o + 0] / landmarkerInputSize;
                // tflite2onnx は Y=0=画像上端（標準画像座標）で出力する。
                // Unity の TextureConverter は Y=0=テクスチャ下端で動作するため反転が必要。
                float ly = 1f - lmSpan[o + 1] / landmarkerInputSize;
                float lz = lmSpan[o + 2];
                Vector2 p = BlazeUtils.ProjectLandmark(lx, ly, in roi, detectorInputSize);
                dst[i] = new FaceKeypoint(p.x, p.y, lz);
            }
        }

        static void BlitRoi(RenderTexture src, RenderTexture dst, in RotatedRect roi, float inputSize)
        {
            float roiSize     = Mathf.Max(roi.Width, roi.Height);
            float roiHalfNorm = roiSize * 0.5f / inputSize;
            float roiCenterU  = roi.CenterX / inputSize;
            float roiCenterV  = 1f - (roi.CenterY / inputSize);

            float u0    = Mathf.Clamp01(roiCenterU - roiHalfNorm);
            float v0    = Mathf.Clamp01(roiCenterV - roiHalfNorm);
            float uSize = Mathf.Min(roiHalfNorm * 2f, 1f - u0);
            float vSize = Mathf.Min(roiHalfNorm * 2f, 1f - v0);

            Graphics.Blit(src, dst, new Vector2(uSize, vSize), new Vector2(u0, v0));
        }

        /// <summary>直近 <see cref="DetectAsync"/> の検出スコア。デバッグ用。</summary>
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
