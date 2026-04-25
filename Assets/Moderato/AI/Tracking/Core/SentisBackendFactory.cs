// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Core/SentisBackendFactory.cs
//
// Sentis (Unity.InferenceEngine) 用バックエンドの選択ヘルパ。
// Windows / macOS スタンドアロンを前提に GPUCompute を最優先し、
// 利用不能な環境では GPUPixel → CPU の順にフォールバックする。

using Unity.InferenceEngine;
using UnityEngine;

namespace Moderato.AI.Tracking.Core
{
    /// <summary>
    /// 実行環境を見て最適な <see cref="BackendType"/> を決める静的ファクトリ。
    /// 各 <see cref="Worker"/> はこの結果を使って構築する。
    /// </summary>
    /// <remarks>
    /// ホットパスでは呼ばれない（起動時 1 回想定）ため、
    /// ここでは GC Alloc の厳密な抑制は気にしなくてよい。
    /// </remarks>
    internal static class SentisBackendFactory
    {
        /// <summary>
        /// このマシンで使うべきバックエンドを返す。
        /// 優先順位：<see cref="BackendType.GPUCompute"/> → <see cref="BackendType.GPUPixel"/> → <see cref="BackendType.CPU"/>。
        /// </summary>
        public static BackendType ResolveBest()
        {
            if (SystemInfo.supportsComputeShaders)
            {
                return BackendType.GPUCompute;
            }

            // ComputeShader が無効でも、GPU 自体はあるなら Pixel パスで動かせる。
            if (SystemInfo.graphicsDeviceType != UnityEngine.Rendering.GraphicsDeviceType.Null)
            {
                return BackendType.GPUPixel;
            }

            return BackendType.CPU;
        }

        /// <summary>
        /// 起動時の診断ログ。3 モダリティ並走中に毎フレーム呼ばないこと。
        /// </summary>
        public static void LogCapabilities(BackendType resolved)
        {
            // string interpolation はホットパス禁止だが、起動時 1 回のみ。
            Debug.Log(
                $"[Moderato.AI.Tracking] Sentis backend: {resolved} " +
                $"(graphicsDevice={SystemInfo.graphicsDeviceType}, " +
                $"supportsCompute={SystemInfo.supportsComputeShaders}, " +
                $"asyncReadback={SystemInfo.supportsAsyncGPUReadback})");
        }
    }
}
