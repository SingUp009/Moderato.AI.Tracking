// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Processor/HandLandmarker.cs
//
// BlazeHand（palm detector + hand landmark）2 段の推論ラッパ。公開 API。
//
// 入力：RenderTexture（典型的には WebcamSource.Frame）
// 出力：最大 2 手分の HandFrame 配列（pre-allocated）
//
// 設計方針：
// - GC Alloc を 0 にするため、Tensor / 出力配列は起動時 1 回だけ確保。
// - 推論完了は Awaitable<Tensor> で待つ（Sentis 純正 readback API）。
// - ReadOnlySpan<T> は async メソッドのローカルにできない (CS4012) ため、
//   Span を触る処理はすべて同期ヘルパに切り出す。
// - 2 手検出は O(2N) 2 パス NMS で 0 alloc 実現（BlazeUtils.ComputeIoU 使用）。
// - ROI クロップは RotatedRect の回転を反映する。top-origin 用の逆射影も同じ回転で整合させる。
//
// 必要な同梱物（Models/ に配置、HuggingFace から手動 DL）：
//   - palm_detection_lite.onnx        : palm detector (input 1×192×192×3)
//   - hand_landmark_lite.onnx         : hand landmark (input 1×224×224×3)
//   - palm_detection_anchors.csv      : ~2016 行の anchor 定義
// 詳細は Models/README.md を参照。

using System;
using System.Threading;
using Moderato.AI.Tracking.Core;
using Unity.InferenceEngine;
using UnityEngine;

namespace Moderato.AI.Tracking.Processor
{
    /// <summary>
    /// BlazeHand 2 段ランドマーカ。最大 2 手を同時検出する。
    /// </summary>
    /// <remarks>
    /// スレッド安全ではない。Update スレッド（メインスレッド）からのみ呼ぶこと。
    /// <see cref="DetectAsync"/> の戻り値の配列は内部バッファの参照。フレームをまたいでキャッシュしないこと。
    /// </remarks>
    public sealed class HandLandmarker : IDisposable
    {
        // ---- palm detector -------------------------------------------------
        const int   k_PalmInputSize  = 192;
        const int   k_PalmBoxStride  = 18;    // cx,cy,w,h, kp0..kp6（各 x,y）
        const int   k_PalmKeypointOffset = 4;
        const int   k_PalmKeypointStride = 2;
        const int   k_PalmWristIndex = 0;
        const int   k_PalmMiddleMcpIndex = 2;

        // ---- hand landmark -------------------------------------------------
        const int   k_LandmarkerSize = 224;
        const int   k_LandmarkCount  = 21;    // TrackingConstants.HandLandmarkCount

        // ---- 検出パラメータ -----------------------------------------------
        const int   k_MaxHands       = 2;     // TrackingConstants.MaxHandCount
        const float k_DetectThresh   = 0.35f; // palm detector のシグモイド閾値。Vサイン等の弱い palm も候補に残す。
        const float k_PresenceThresh = 0.5f;  // landmark model presence 閾値
        const float k_NmsIouThresh   = 0.5f;  // NMS IoU 閾値（両手が隣接しても検出できるよう緩和）
        // BlazeHand の ROI スケール。MediaPipe 標準は 2.6（BlazePose の 1.25 より大きい）。
        const float k_RoiScale       = 2.6f;
        const float k_AxisFallbackScale = 1.15f;
        const string k_RoiShaderResource = "Moderato.AI.Tracking/RotatedRoiBlit";
        const string k_RoiShaderName = "Hidden/Moderato/AI/Tracking/RotatedRoiBlit";

        static readonly int s_RoiShaderRoiId       = Shader.PropertyToID("_Roi");
        static readonly int s_RoiShaderInputSizeId = Shader.PropertyToID("_InputSize");
        static readonly int s_RoiShaderRotationId  = Shader.PropertyToID("_RoiRotation");

        readonly Worker m_PalmWorker;
        readonly Worker m_LandmarkerWorker;

        readonly Tensor<float> m_PalmInput;       // (1, 192, 192, 3)
        readonly Tensor<float> m_LandmarkerInput; // (1, 224, 224, 3)
        readonly RenderTexture m_RoiTexture;      // 224×224 共有（2 手分を逐次使い回す）
        readonly Material m_RoiMaterial;           // 回転 ROI クロップ用（起動時 1 回だけ確保）

        readonly PoseAnchor[] m_Anchors;

        readonly HandKeypoint[] m_Landmarks0; // 21 点 hand 0
        readonly HandKeypoint[] m_Landmarks1; // 21 点 hand 1
        readonly HandFrame[]    m_Result;     // [2]、DetectAsync の戻り値

        readonly TextureTransform m_PalmTransform;
        readonly TextureTransform m_LandmarkerTransform;

        int  m_DetectedHandCount;
        bool m_Disposed;

        readonly struct HandRoi
        {
            public readonly RotatedRect Rect;
            public readonly RotatedRect AxisRect;
            public readonly float WristX;
            public readonly float WristY;
            public readonly float MiddleMcpX;
            public readonly float MiddleMcpY;

            public HandRoi(
                RotatedRect rect,
                RotatedRect axisRect,
                float wristX,
                float wristY,
                float middleMcpX,
                float middleMcpY)
            {
                Rect       = rect;
                AxisRect   = axisRect;
                WristX     = wristX;
                WristY     = wristY;
                MiddleMcpX = middleMcpX;
                MiddleMcpY = middleMcpY;
            }

            public HandRoi WithRect(RotatedRect rect)
            {
                return new HandRoi(rect, AxisRect, WristX, WristY, MiddleMcpX, MiddleMcpY);
            }
        }

        /// <summary>
        /// 各モデル・anchor を渡して構築する。
        /// </summary>
        /// <param name="palmDetector">Palm detector の <see cref="ModelAsset"/>。</param>
        /// <param name="handLandmarker">Hand landmark モデルの <see cref="ModelAsset"/>。</param>
        /// <param name="palmAnchorsCsv">palm_detection_anchors.csv の <see cref="TextAsset"/>（~2016 行）。</param>
        /// <param name="backend">通常は <see cref="SentisBackendFactory.ResolveBest"/> の戻り値。</param>
        public HandLandmarker(
            ModelAsset  palmDetector,
            ModelAsset  handLandmarker,
            TextAsset   palmAnchorsCsv,
            BackendType backend)
        {
            if (palmDetector   == null) throw new ArgumentNullException(nameof(palmDetector));
            if (handLandmarker == null) throw new ArgumentNullException(nameof(handLandmarker));
            if (palmAnchorsCsv == null) throw new ArgumentNullException(nameof(palmAnchorsCsv));

            var palmModel      = ModelLoader.Load(palmDetector);
            var landmarkerModel = ModelLoader.Load(handLandmarker);

            m_PalmWorker      = new Worker(palmModel,       backend);
            m_LandmarkerWorker = new Worker(landmarkerModel, backend);

            m_PalmInput       = new Tensor<float>(new TensorShape(1, k_PalmInputSize,  k_PalmInputSize,  3));
            m_LandmarkerInput = new Tensor<float>(new TensorShape(1, k_LandmarkerSize, k_LandmarkerSize, 3));

            m_RoiTexture = new RenderTexture(k_LandmarkerSize, k_LandmarkerSize, 0, RenderTextureFormat.ARGB32)
            {
                name       = "Moderato.AI.Tracking.HandRoi",
                wrapMode   = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
            };
            m_RoiTexture.Create();

            var roiShader = Resources.Load<Shader>(k_RoiShaderResource);
            if (roiShader == null)
                roiShader = Shader.Find(k_RoiShaderName);

            if (roiShader != null)
            {
                m_RoiMaterial = new Material(roiShader)
                {
                    hideFlags = HideFlags.HideAndDontSave,
                };
            }
            else
            {
                Debug.LogWarning(
                    "[Moderato.AI.Tracking] RotatedRoiBlit shader not found. " +
                    "Hand ROI crop will fall back to axis-aligned mode.");
            }

            m_Anchors = PoseAnchors.Load(palmAnchorsCsv.text);
            if (Mathf.Abs(m_Anchors.Length - 2016) > 100)
            {
                Debug.LogWarning(
                    $"[Moderato.AI.Tracking] Palm anchor count={m_Anchors.Length}, expected ~2016. " +
                    "Verify that palm_detection_anchors.csv matches the palm detector model.");
            }

            m_Landmarks0 = new HandKeypoint[k_LandmarkCount];
            m_Landmarks1 = new HandKeypoint[k_LandmarkCount];
            m_Result     = new HandFrame[k_MaxHands];

            // BlazePose / BlazeHand ともに NHWC。SetTensorLayout(N=0, C=3, H=1, W=2)。
            m_PalmTransform = new TextureTransform()
                .SetDimensions(k_PalmInputSize, k_PalmInputSize, 3)
                .SetTensorLayout(0, 3, 1, 2);
            m_LandmarkerTransform = new TextureTransform()
                .SetDimensions(k_LandmarkerSize, k_LandmarkerSize, 3)
                .SetTensorLayout(0, 3, 1, 2);
        }

        /// <summary>
        /// 1 フレームの推論を実行し、最大 2 手の検出結果を返す。
        /// </summary>
        /// <param name="frame">入力フレーム。WebcamSource.Frame をそのまま渡す想定。</param>
        /// <param name="cancellationToken">中断トークン（任意）。</param>
        /// <returns>
        /// 長さ 2 の <see cref="HandFrame"/> 配列。インデックス 0/1 が各手に対応する。
        /// 検出されなかったスロットは <see cref="HandFrame.IsValid"/> == false。
        /// 配列は内部バッファの参照。フレームをまたいでキャッシュしないこと。
        /// </returns>
        public async Awaitable<HandFrame[]> DetectAsync(
            RenderTexture frame,
            CancellationToken cancellationToken = default)
        {
            if (m_Disposed) throw new ObjectDisposedException(nameof(HandLandmarker));
            if (frame == null) throw new ArgumentNullException(nameof(frame));

            cancellationToken.ThrowIfCancellationRequested();

            // ---- 1) 入力 RT → palm detector Tensor（GPU 直） ----
            TextureConverter.ToTensor(frame, m_PalmInput, m_PalmTransform);

            // ---- 2) palm detector を Schedule ----
            m_PalmWorker.Schedule(m_PalmInput);

            // ---- 3) 非同期 readback ----
            using var boxesCpu = (Tensor<float>)await m_PalmWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using var scoresCpu = (Tensor<float>)await m_PalmWorker.PeekOutput(1).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            // ---- 4) CPU 側で 2 手 NMS ----
            int handCount = TryFindTopPalms(
                scoresCpu, boxesCpu, m_Anchors, k_PalmInputSize,
                k_DetectThresh, k_NmsIouThresh,
                out float score0, out HandRoi roi0,
                out float score1, out HandRoi roi1);

            m_DetectedHandCount = handCount;

            // 結果スロットをリセット
            m_Result[0] = new HandFrame(m_Landmarks0, Handedness.Unknown, 0f, 0f, false);
            m_Result[1] = new HandFrame(m_Landmarks1, Handedness.Unknown, 0f, 0f, false);

            if (handCount == 0)
                return m_Result;

            // ---- 5) hand 0 の landmark 推論 ----
            m_Result[0] = await RunHandLandmarkAsync(
                frame, roi0, roi0.Rect,
                useRotatedCrop: true,
                score0, cancellationToken, m_Landmarks0);

            if (!m_Result[0].IsValid)
            {
                HandFrame fallback = await RunHandLandmarkAsync(
                    frame, roi0, roi0.AxisRect,
                    useRotatedCrop: false,
                    score0, cancellationToken, m_Landmarks0);

                if (fallback.PresenceScore >= m_Result[0].PresenceScore)
                    m_Result[0] = fallback;
            }

            if (handCount < 2)
                return m_Result;

            // ---- 6) hand 1 の landmark 推論（ROI テクスチャ上書き、readback 済みなので安全） ----
            m_Result[1] = await RunHandLandmarkAsync(
                frame, roi1, roi1.Rect,
                useRotatedCrop: true,
                score1, cancellationToken, m_Landmarks1);

            if (!m_Result[1].IsValid)
            {
                HandFrame fallback = await RunHandLandmarkAsync(
                    frame, roi1, roi1.AxisRect,
                    useRotatedCrop: false,
                    score1, cancellationToken, m_Landmarks1);

                if (fallback.PresenceScore >= m_Result[1].PresenceScore)
                    m_Result[1] = fallback;
            }

            SortHandsByScreenX(m_Result);
            return m_Result;
        }

        async Awaitable<HandFrame> RunHandLandmarkAsync(
            RenderTexture frame,
            HandRoi roi,
            RotatedRect cropRect,
            bool useRotatedCrop,
            float palmDetectionScore,
            CancellationToken cancellationToken,
            HandKeypoint[] dst)
        {
            bool useRoiRotation;
            if (useRotatedCrop)
            {
                useRoiRotation = BlitRoi(frame, m_RoiTexture, in cropRect, k_PalmInputSize, m_RoiMaterial);
            }
            else
            {
                BlitAxisAlignedRoi(frame, m_RoiTexture, in cropRect, k_PalmInputSize);
                useRoiRotation = false;
            }

            TextureConverter.ToTensor(m_RoiTexture, m_LandmarkerInput, m_LandmarkerTransform);
            m_LandmarkerWorker.Schedule(m_LandmarkerInput);

            using var lm = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(0).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            // tf2onnx 変換後: Index1=presence(hand_flag), Index2=handedness, Index3=world_landmarks(未使用)
            using var pres = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(1).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();
            using var hand = (Tensor<float>)await m_LandmarkerWorker.PeekOutput(2).ReadbackAndCloneAsync();
            cancellationToken.ThrowIfCancellationRequested();

            HandRoi projectionRoi = roi.WithRect(cropRect);
            return DecodeLandmarkResult(
                lm, hand, pres, in projectionRoi,
                k_PalmInputSize, k_LandmarkerSize,
                palmDetectionScore, k_PresenceThresh, useRoiRotation, dst);
        }

        // ----------------------------------------------------------------------
        // 同期ヘルパ群（ReadOnlySpan を扱う処理を async メソッドから切り出す）
        // すべて static。副作用は out 引数経由のみ。
        // ----------------------------------------------------------------------

        /// <summary>
        /// Palm detector 出力から上位 2 手を NMS で選ぶ。
        /// </summary>
        /// <returns>検出手数（0, 1, 2 のいずれか）。</returns>
        static int TryFindTopPalms(
            Tensor<float> scores,
            Tensor<float> boxes,
            PoseAnchor[]  anchors,
            float         inputSize,
            float         detectThresh,
            float         iouThresh,
            out float      score0, out HandRoi roi0,
            out float      score1, out HandRoi roi1)
        {
            ReadOnlySpan<float> scoresSpan = scores.AsReadOnlySpan();
            ReadOnlySpan<float> boxesSpan  = boxes.AsReadOnlySpan();
            int anchorCount = anchors.Length;

            // --- Pass 1: 最高スコアの hand を探す ---
            float best1Raw = float.NegativeInfinity;
            int   best1Idx = -1;

            for (int i = 0; i < anchorCount; i++)
            {
                float raw = scoresSpan[i];
                if (raw > best1Raw)
                {
                    best1Raw = raw;
                    best1Idx = i;
                }
            }

            float best1Sig = BlazeUtils.Sigmoid(best1Raw);
            if (best1Idx < 0 || best1Idx >= anchorCount || best1Sig < detectThresh)
            {
                score0 = 0f; roi0 = default;
                score1 = 0f; roi1 = default;
                return 0;
            }

            score0 = best1Sig;
            roi0   = DecodeHandRoi(boxesSpan, anchors, best1Idx, inputSize);

            // --- Pass 2: IoU が閾値未満の次点を探す ---
            float best2Raw = float.NegativeInfinity;
            int   best2Idx = -1;
            roi1 = default;

            for (int i = 0; i < anchorCount; i++)
            {
                if (i == best1Idx) continue;

                float raw = scoresSpan[i];
                float sig = BlazeUtils.Sigmoid(raw);
                if (sig < detectThresh) continue;

                HandRoi candidate = DecodeHandRoi(boxesSpan, anchors, i, inputSize);
                if (BlazeUtils.ComputeIoU(in roi0.Rect, in candidate.Rect) >= iouThresh) continue;

                if (raw > best2Raw)
                {
                    best2Raw = raw;
                    best2Idx = i;
                    roi1     = candidate;
                }
            }

            if (best2Idx < 0)
            {
                score1 = 0f;
                return 1;
            }

            score1 = BlazeUtils.Sigmoid(best2Raw);
            return 2;
        }

        /// <summary>
        /// boxes スパンの i 番目エントリを <see cref="RotatedRect"/> に変換する。
        /// </summary>
        static HandRoi DecodeHandRoi(
            ReadOnlySpan<float> boxesSpan,
            PoseAnchor[]        anchors,
            int                 idx,
            float               inputSize)
        {
            // boxes shape: (1, N, 18) — Slice でストライド分だけ取り出す。
            ReadOnlySpan<float> raw = boxesSpan.Slice(idx * k_PalmBoxStride, k_PalmBoxStride);
            var anchor = anchors[idx];

            BlazeUtils.DecodeBox(
                raw,
                anchor.CenterX, anchor.CenterY,
                inputSize,
                out float boxCx, out float boxCy,
                out float boxW, out float boxH,
                out _, out _,
                out _, out _);

            int wristOffset = k_PalmKeypointOffset + k_PalmWristIndex * k_PalmKeypointStride;
            float wristX = raw[wristOffset + 0] + anchor.CenterX * inputSize;
            float wristY = raw[wristOffset + 1] + anchor.CenterY * inputSize;
            int middleMcpOffset = k_PalmKeypointOffset + k_PalmMiddleMcpIndex * k_PalmKeypointStride;
            float middleMcpX = raw[middleMcpOffset + 0] + anchor.CenterX * inputSize;
            float middleMcpY = raw[middleMcpOffset + 1] + anchor.CenterY * inputSize;

            // Unity BlazeHand サンプルと同じく、palm box の長辺を 2.6 倍し、
            // 中心を wrist→middle MCP 方向へ 0.5*boxSize だけずらして指先側を含める。
            float boxSize = Mathf.Max(boxW, boxH);
            RotatedRect rect = BlazeUtils.MakeHandRoi(
                boxCx, boxCy, boxSize, wristX, wristY, middleMcpX, middleMcpY, k_RoiScale);

            // 回転ROIが landmarker の presence を落とす場合に備え、
            // 少し広めの軸沿いROIを診断兼フォールバックとして保持する。
            float axisSide = rect.Width * k_AxisFallbackScale;
            var axisRect = new RotatedRect(rect.CenterX, rect.CenterY, axisSide, axisSide, 0f);

            return new HandRoi(rect, axisRect, wristX, wristY, middleMcpX, middleMcpY);
        }

        /// <summary>
        /// landmark モデルの出力テンソル 3 本を 1 手の <see cref="HandFrame"/> に変換する。
        /// </summary>
        static HandFrame DecodeLandmarkResult(
            Tensor<float> landmarks,
            Tensor<float> handedness,
            Tensor<float> presence,
            in HandRoi roi,
            float          detectorInputSize,
            float          landmarkerInputSize,
            float          palmDetectionScore,
            float          presenceThresh,
            bool           useRoiRotation,
            HandKeypoint[] dst)
        {
            // shape アサート（出力インデックスのずれを早期検出）
            Debug.Assert(landmarks.shape[1] == 63,
                $"[Moderato.AI.Tracking] hand landmark output shape mismatch: shape[1]={landmarks.shape[1]}, expected 63");
            Debug.Assert(handedness.shape[1] == 1,
                $"[Moderato.AI.Tracking] handedness output shape mismatch: shape[1]={handedness.shape[1]}, expected 1");
            Debug.Assert(presence.shape[1] == 1,
                $"[Moderato.AI.Tracking] presence output shape mismatch: shape[1]={presence.shape[1]}, expected 1");

            float presenceScore = presence.AsReadOnlySpan()[0];

            if (presenceScore < presenceThresh)
                return new HandFrame(dst, Handedness.Unknown, palmDetectionScore, presenceScore, false);

            // handedness: BlazeHand の出力は sigmoid 後の [0,1] スコア。カメラ視点で判定。
            // 未ミラーの WebCamTexture（カメラの右 = ユーザーの左）では Left/Right が反転するため
            // ここでユーザー視点に変換する。[0.3, 0.7] の曖昧域は Unknown とし毎フレームの揺れを防ぐ。
            float handednessRaw = handedness.AsReadOnlySpan()[0];
            Handedness hand;
            if      (handednessRaw > 0.7f) hand = Handedness.Left;    // カメラ右(>0.7) = ユーザー左
            else if (handednessRaw < 0.3f) hand = Handedness.Right;   // カメラ左(<0.3) = ユーザー右
            else                           hand = Handedness.Unknown;  // 曖昧域はスキップ

            // 回転 ROI クロップが有効なときは同じ rotation で逆射影する。
            // 軸沿いフォールバックは Graphics.Blit の矩形クロップなので、Y を追加反転しない射影を使う。
            RotatedRect projectionRoi = useRoiRotation
                ? roi.Rect
                : new RotatedRect(roi.Rect.CenterX, roi.Rect.CenterY, roi.Rect.Width, roi.Rect.Height, 0f);

            ReadOnlySpan<float> lmSpan = landmarks.AsReadOnlySpan();
            for (int i = 0; i < k_LandmarkCount; i++)
            {
                int o = i * 3;
                float lx = lmSpan[o + 0] / landmarkerInputSize;
                float ly = lmSpan[o + 1] / landmarkerInputSize;
                float lz = lmSpan[o + 2]; // 深度はピクセル単位のまま保持

                Vector2 projected = useRoiRotation
                    ? BlazeUtils.ProjectLandmarkTopOrigin(lx, ly, in projectionRoi, detectorInputSize)
                    : BlazeUtils.ProjectLandmark(lx, ly, in projectionRoi, detectorInputSize);
                dst[i] = new HandKeypoint(projected.x, projected.y, lz);
            }

            return new HandFrame(dst, hand, palmDetectionScore, presenceScore, true);
        }

        static void SortHandsByScreenX(HandFrame[] hands)
        {
            if (hands == null || hands.Length < 2) return;
            if (!hands[0].IsValid || !hands[1].IsValid) return;

            float x0 = hands[0].Landmarks[0].X;
            float x1 = hands[1].Landmarks[0].X;
            if (x0 <= x1) return;

            (hands[0], hands[1]) = (hands[1], hands[0]);
        }

        /// <summary>
        /// 検出された ROI を src から dst に回転クロップする（Blit）。
        /// シェーダが利用できない場合のみ軸沿いクロップへフォールバックする。
        /// </summary>
        /// <returns>回転クロップを適用できた場合 true。</returns>
        static bool BlitRoi(
            RenderTexture src,
            RenderTexture dst,
            in RotatedRect roi,
            float inputSize,
            Material roiMaterial)
        {
            if (roiMaterial == null)
            {
                BlitAxisAlignedRoi(src, dst, in roi, inputSize);
                return false;
            }

            roiMaterial.SetVector(s_RoiShaderRoiId,
                new Vector4(roi.CenterX, roi.CenterY, roi.Width, roi.Height));
            roiMaterial.SetFloat(s_RoiShaderInputSizeId, inputSize);
            roiMaterial.SetFloat(s_RoiShaderRotationId, roi.Rotation);
            Graphics.Blit(src, dst, roiMaterial);
            return true;
        }

        static void BlitAxisAlignedRoi(RenderTexture src, RenderTexture dst, in RotatedRect roi, float inputSize)
        {
            float roiSize      = Mathf.Max(roi.Width, roi.Height);
            float roiHalfNorm  = (roiSize * 0.5f) / inputSize;
            float roiCenterU   = roi.CenterX / inputSize;
            float roiCenterV   = 1f - (roi.CenterY / inputSize); // Blit の V は下原点

            float u0    = Mathf.Clamp01(roiCenterU - roiHalfNorm);
            float v0    = Mathf.Clamp01(roiCenterV - roiHalfNorm);
            float uSize = Mathf.Min(roiHalfNorm * 2f, 1f - u0);
            float vSize = Mathf.Min(roiHalfNorm * 2f, 1f - v0);

            Graphics.Blit(src, dst,
                new Vector2(uSize, vSize),
                new Vector2(u0, v0));
        }

        /// <summary>直近フレームで検出した手の数（0〜2）。デバッグ / Profiler 用。</summary>
        public int DetectedHandCount => m_DetectedHandCount;

        /// <summary>直近の hand landmarker 入力 ROI。Sandbox 表示用。</summary>
        public Texture DebugLastRoiTexture => m_RoiTexture;

        public void Dispose()
        {
            if (m_Disposed) return;
            m_Disposed = true;

            m_PalmWorker?.Dispose();
            m_LandmarkerWorker?.Dispose();
            m_PalmInput?.Dispose();
            m_LandmarkerInput?.Dispose();

            if (m_RoiMaterial != null)
                UnityEngine.Object.Destroy(m_RoiMaterial);

            if (m_RoiTexture != null)
            {
                m_RoiTexture.Release();
                UnityEngine.Object.Destroy(m_RoiTexture);
            }
        }
    }
}
