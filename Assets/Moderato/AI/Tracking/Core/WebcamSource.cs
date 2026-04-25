// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Core/WebcamSource.cs
//
// Web カメラから固定サイズの RenderTexture へ毎フレーム転送する内部ユーティリティ。
// Sentis の TextureConverter.ToTensor は RenderTexture を直接食えるので、CPU 経由のコピーは作らない。
//
// 設計方針：
// - MonoBehaviour ではなく素の class。呼び出し側（Processor/ 配下のクラス）が
//   Update のタイミングで Pump() を呼ぶ。これによりパッケージ消費側が Component を
//   貼る必要がなく、テストも書きやすい。
// - 起動時に WebCamTexture と RenderTexture を 1 つずつ確保し、以降は使い回す（0 alloc）。
// - 取得失敗時は false を返すだけで例外は投げない。3 モデル並走中の中断耐性を優先。

using System;
using UnityEngine;

namespace Moderato.AI.Tracking.Core
{
    /// <summary>
    /// Web カメラ入力を Sentis が扱える <see cref="RenderTexture"/> に整える内部ヘルパ。
    /// </summary>
    internal sealed class WebcamSource : IDisposable
    {
        readonly int m_TargetWidth;
        readonly int m_TargetHeight;
        readonly int m_RequestedFps;
        readonly string m_RequestedDeviceName;

        WebCamTexture m_Webcam;
        RenderTexture m_Frame;
        int m_LastUpdatedFrame = -1;
        bool m_Disposed;

        /// <summary>
        /// 直近に内容更新された RenderTexture。<see cref="Pump"/> 後に <see cref="HasFreshFrame"/> が
        /// true のときだけ参照すること。
        /// </summary>
        public RenderTexture Frame => m_Frame;

        /// <summary>
        /// 今フレームに新しい Web カメラ画像を反映したか。<see cref="Pump"/> 呼出後に評価する。
        /// </summary>
        public bool HasFreshFrame { get; private set; }

        public bool IsRunning => m_Webcam != null && m_Webcam.isPlaying;

        /// <summary>
        /// 出力 RenderTexture の固定サイズ・希望 FPS・カメラ名を指定して作成する。
        /// カメラ名 null の場合は OS 既定デバイスを使う。
        /// </summary>
        public WebcamSource(int width, int height, int requestedFps = 30, string deviceName = null)
        {
            if (width <= 0 || height <= 0)
                throw new ArgumentOutOfRangeException(nameof(width), "width/height must be positive.");

            m_TargetWidth = width;
            m_TargetHeight = height;
            m_RequestedFps = requestedFps;
            m_RequestedDeviceName = deviceName;
        }

        /// <summary>
        /// Web カメラを起動し、固定サイズの RenderTexture を確保する。多重呼出は無視。
        /// </summary>
        public void Start()
        {
            if (m_Disposed) throw new ObjectDisposedException(nameof(WebcamSource));
            if (m_Webcam != null) return;

            // RenderTexture は最終出力サイズに合わせて 1 本確保し使い回す。
            // 8bit RGBA で十分（Sentis は ToTensor 内部で正規化する）。
            m_Frame = new RenderTexture(m_TargetWidth, m_TargetHeight, 0, RenderTextureFormat.ARGB32)
            {
                name = "Moderato.AI.Tracking.WebcamFrame",
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear,
                useMipMap = false,
                autoGenerateMips = false,
            };
            m_Frame.Create();

            // デバイス選択：指定されていなければ OS 既定。
            string device = m_RequestedDeviceName;
            if (string.IsNullOrEmpty(device) && WebCamTexture.devices.Length > 0)
            {
                device = WebCamTexture.devices[0].name;
            }

            m_Webcam = string.IsNullOrEmpty(device)
                ? new WebCamTexture(m_TargetWidth, m_TargetHeight, m_RequestedFps)
                : new WebCamTexture(device, m_TargetWidth, m_TargetHeight, m_RequestedFps);

            m_Webcam.Play();
        }

        /// <summary>
        /// 毎フレーム呼ぶ。Web カメラに新フレームが届いていれば <see cref="Frame"/> に Blit する。
        /// 更新があれば <see cref="HasFreshFrame"/> が true になる。
        /// </summary>
        /// <remarks>
        /// アロケーションを出さないため、ここでは <c>$"..."</c> や <c>List&lt;T&gt;</c> を使わない。
        /// </remarks>
        public void Pump()
        {
            HasFreshFrame = false;

            if (!IsRunning) return;

            // didUpdateThisFrame は同フレーム内の重複 Pump を防ぐ。
            // m_LastUpdatedFrame は外側からの再呼出に対するガード。
            if (!m_Webcam.didUpdateThisFrame) return;
            if (m_LastUpdatedFrame == Time.frameCount) return;

            // Webcam → RenderTexture のサイズ違い・回転は Blit が解決する。
            // 回転（OS 縦向き等）が必要な場合は今後 TextureTransform で対応。
            Graphics.Blit(m_Webcam, m_Frame);

            m_LastUpdatedFrame = Time.frameCount;
            HasFreshFrame = true;
        }

        public void Stop()
        {
            if (m_Webcam != null && m_Webcam.isPlaying)
            {
                m_Webcam.Stop();
            }
        }

        public void Dispose()
        {
            if (m_Disposed) return;
            m_Disposed = true;

            Stop();

            if (m_Webcam != null)
            {
                UnityEngine.Object.Destroy(m_Webcam);
                m_Webcam = null;
            }

            if (m_Frame != null)
            {
                m_Frame.Release();
                UnityEngine.Object.Destroy(m_Frame);
                m_Frame = null;
            }
        }
    }
}
