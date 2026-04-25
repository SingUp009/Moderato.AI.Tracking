// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Core/BlazeUtils.cs
//
// BlazePose / BlazeFace / BlazeHand 共通の幾何ユーティリティ。
// MediaPipe / Unity 公式 inference-engine-samples の BlazeUtils.cs を参考に、
// このプロジェクトの 0 alloc 制約に合わせ struct ベースで再実装している。
//
// 提供するもの：
// - 正規化座標と画素座標の往復
// - 検出器の生出力（sigmoid 前 score, 中心+幅高+回転キーポイント）の decode
// - 2 点（典型的にはアンカーキーポイント）から RotatedRect を作る
// - RotatedRect → 2D アフィン行列（landmark 入力用 ROI クロップ用）
// - 出力ランドマーク（landmarker ローカル座標）→ 入力画像座標 への射影
//
// すべて static method で副作用なし。Tensor は触らない（呼び出し側で readback 済み配列を渡す）。

using System;
using UnityEngine;

namespace Moderato.AI.Tracking.Core
{
    /// <summary>
    /// 中心点・幅高・回転角を持つ矩形。BlazePose では「腰中心 → 肩中心」の方向で回転を取る。
    /// </summary>
    internal readonly struct RotatedRect
    {
        public readonly float CenterX;
        public readonly float CenterY;
        public readonly float Width;
        public readonly float Height;
        /// <summary>ラジアン。MediaPipe 仕様では「上向き = -π/2」基準。</summary>
        public readonly float Rotation;

        public RotatedRect(float centerX, float centerY, float width, float height, float rotation)
        {
            CenterX = centerX;
            CenterY = centerY;
            Width = width;
            Height = height;
            Rotation = rotation;
        }
    }

    /// <summary>
    /// BlazePose 系で繰り返し使う幾何ヘルパ。すべて static / 値型。
    /// </summary>
    internal static class BlazeUtils
    {
        /// <summary>シグモイド。Mathf.Exp は内部で Allocation しないので OK。</summary>
        public static float Sigmoid(float x)
        {
            // overflow ガード（MediaPipe の clipping と同等）
            if (x < -50f) return 0f;
            if (x > 50f) return 1f;
            return 1f / (1f + Mathf.Exp(-x));
        }

        /// <summary>
        /// BlazePose detector の anchor 1 本に対する box decode。
        /// <paramref name="raw"/> は (12,) ベクトル：cx,cy,w,h,kp0x,kp0y,...,kp3x,kp3y（offset 単位、anchor 相対）。
        /// 戻り値はピクセル空間ではなく **入力画像 (0..inputSize-1)** の値。
        /// </summary>
        public static void DecodeBox(
            ReadOnlySpan<float> raw,
            float anchorCenterX, float anchorCenterY,
            float inputSize,
            out float boxCenterX, out float boxCenterY,
            out float boxWidth, out float boxHeight,
            out float midHipX, out float midHipY,
            out float fullBodyX, out float fullBodyY)
        {
            // BlazePose detector の出力は「入力画像のピクセル単位」で
            // anchor 中心に対するオフセット。スケール (=inputSize) は適用済みのまま使う。
            boxCenterX = raw[0] + anchorCenterX * inputSize;
            boxCenterY = raw[1] + anchorCenterY * inputSize;
            boxWidth = raw[2];
            boxHeight = raw[3];

            // keypoint 0 = mid-hip, 1 = full body center（モデル仕様）。
            midHipX = raw[4] + anchorCenterX * inputSize;
            midHipY = raw[5] + anchorCenterY * inputSize;
            fullBodyX = raw[6] + anchorCenterX * inputSize;
            fullBodyY = raw[7] + anchorCenterY * inputSize;
        }

        /// <summary>
        /// detector が出した「腰中心 (kp0)」「全身中心 (kp1)」から
        /// landmarker 入力用の RotatedRect（正方形・MediaPipe スケール）を作る。
        /// </summary>
        /// <remarks>
        /// MediaPipe Pose の AlignmentPointsRectsCalculator 相当。
        /// 上下左右に 1.25 倍のマージンを取り、回転は 2 点を結ぶベクトルから計算。
        /// </remarks>
        public static RotatedRect MakeRoi(
            float midHipX, float midHipY,
            float fullBodyX, float fullBodyY,
            float scaleFactor = 1.25f)
        {
            float dx = fullBodyX - midHipX;
            float dy = fullBodyY - midHipY;

            // MediaPipe の rotation：「上向き(-Y方向)が 0 ラジアン」基準。
            // C# の Atan2 は (y,x) で右が 0 なので π/2 を加減する。
            float rotation = -(Mathf.Atan2(-dy, dx)) - Mathf.PI * 0.5f;

            float distance = Mathf.Sqrt(dx * dx + dy * dy);
            // 体の縦半径 = 2 点間距離。スケール 1.25 で軽くマージン。
            float side = 2f * distance * scaleFactor;

            return new RotatedRect(midHipX, midHipY, side, side, rotation);
        }

        /// <summary>
        /// landmarker のローカル正規化座標 (0..1, 0..1) を、入力画像の正規化座標 (0..1) に戻す。
        /// </summary>
        /// <remarks>
        /// landmarker は <paramref name="roi"/> をクロップ・回転した小画像で訓練されている。
        /// 出力 (lx,ly) は「クロップ画像の左上 (0,0) 〜右下 (1,1)」基準。
        /// これを元の入力画像（幅高 <paramref name="inputSize"/> ピクセル）の正規化座標に戻す。
        /// </remarks>
        public static Vector2 ProjectLandmark(
            float lx, float ly,
            in RotatedRect roi,
            float inputSize)
        {
            // クロップ画像中心を 0,0 にして、(-0.5, 0.5) のローカル座標に変換
            float cx = lx - 0.5f;
            float cy = ly - 0.5f;

            // ROI の幅高（ピクセル単位）でスケール
            float sx = cx * roi.Width;
            float sy = cy * roi.Height;

            // 回転を入れる（ROI の Rotation を逆方向で適用しない＝順方向に回す）
            float cos = Mathf.Cos(roi.Rotation);
            float sin = Mathf.Sin(roi.Rotation);
            float rx = sx * cos - sy * sin;
            float ry = sx * sin + sy * cos;

            // ROI 中心へ加算してピクセル座標、最後に inputSize で正規化
            float px = (roi.CenterX + rx) / inputSize;
            float py = (roi.CenterY + ry) / inputSize;
            return new Vector2(px, py);
        }
    }
}
