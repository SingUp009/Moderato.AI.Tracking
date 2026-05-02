// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Processor/TrackingService.cs
//
// M8: PoseLandmarker / HandLandmarker / FaceLandmarker を束ねる統合サービス。
//
// GPU パイプライン戦略：
//   各 DetectAsync は最初の await（Detector readback）まで同期実行される。
//   3 行の呼び出しで全 Detector GPU ジョブがフレーム内に先行投入され、
//   GPU キューに 3 本のパイプラインが並列スケジュールされる。

using System;
using System.Threading;
using Unity.InferenceEngine;
using UnityEngine;

namespace Moderato.AI.Tracking.Processor
{
    /// <summary>
    /// ポーズ・手・顔の 3 モダリティを 1 つの API で提供する統合トラッカー。
    /// </summary>
    /// <remarks>
    /// スレッド安全ではない。メインスレッドからのみ呼ぶこと。
    /// <see cref="DetectAsync"/> の結果に含まれる配列は各 Processor 内部バッファの参照。
    /// フレームをまたいでキャッシュしないこと。
    /// </remarks>
    public sealed class TrackingService : IDisposable
    {
        readonly PoseLandmarker m_Pose;
        readonly HandLandmarker m_Hand;
        readonly FaceLandmarker m_Face;
        bool m_Disposed;

        /// <summary>
        /// 各モデル・anchor アセットを渡して構築する。
        /// </summary>
        public TrackingService(
            ModelAsset poseDetector,  ModelAsset poseLandmarker, TextAsset poseAnchors,
            ModelAsset palmDetector,  ModelAsset handLandmarker, TextAsset palmAnchors,
            ModelAsset faceDetector,  ModelAsset faceLandmarker, TextAsset faceAnchors,
            BackendType backend)
        {
            m_Pose = new PoseLandmarker(poseDetector, poseLandmarker, poseAnchors, backend);
            m_Hand = new HandLandmarker(palmDetector, handLandmarker, palmAnchors, backend);
            m_Face = new FaceLandmarker(faceDetector, faceLandmarker, faceAnchors, backend);
        }

        /// <summary>
        /// 1 フレームの 3 モダリティ推論を実行する。
        /// </summary>
        /// <param name="frame">入力フレーム（WebcamSource.Frame をそのまま渡す想定）。</param>
        /// <param name="cancellationToken">中断トークン（任意）。</param>
        /// <returns>3 モダリティ分の <see cref="TrackingFrame"/>。含まれる配列は内部バッファの参照。</returns>
        public async Awaitable<TrackingFrame> DetectAsync(
            RenderTexture frame,
            CancellationToken cancellationToken = default)
        {
            if (m_Disposed) throw new ObjectDisposedException(nameof(TrackingService));
            if (frame == null) throw new ArgumentNullException(nameof(frame));

            // 3 Detector GPU ジョブを先行投入（各 DetectAsync は最初の await まで同期実行）
            var poseAwaitable  = m_Pose.DetectAsync(frame, cancellationToken);
            var handsAwaitable = m_Hand.DetectAsync(frame, cancellationToken);
            var faceAwaitable  = m_Face.DetectAsync(frame, cancellationToken);

            var pose  = await poseAwaitable;
            var hands = await handsAwaitable;
            var face  = await faceAwaitable;

            return new TrackingFrame(pose, hands, face);
        }

        /// <summary>直近の hand landmarker 入力 ROI。Sandbox 表示用。</summary>
        public Texture DebugHandRoiTexture => m_Hand.DebugLastRoiTexture;

        public void Dispose()
        {
            if (m_Disposed) return;
            m_Disposed = true;

            m_Pose?.Dispose();
            m_Hand?.Dispose();
            m_Face?.Dispose();
        }
    }
}
