// SPDX-License-Identifier: MIT
// Moderato.AI.Tracking — Core/PoseAnchors.cs
//
// BlazePose detector の anchor を CSV TextAsset から起動時 1 回だけ読み込むユーティリティ。
//
// CSV フォーマット（Unity 公式 unity/sentis-blaze-pose の anchors.csv に揃える）：
//   1 行 = 1 anchor、カラム = "centerX,centerY,width,height"
//   行数は 2254（detector の出力 anchor 数と一致）
//
// 設計方針：
// - 起動時 1 回のロード（Worker 構築前）。Update 中は触らない。
// - 戻すのは float[] x 4（X, Y, W, H をストライプ化）して cache 局所性を上げる。
// - CSV パースで string.Split を使うのは起動時のみなので Allocation は許容。

using System;
using UnityEngine;

namespace Moderato.AI.Tracking.Core
{
    /// <summary>
    /// BlazePose detector の anchor。値型ペアで保持。
    /// </summary>
    internal readonly struct PoseAnchor
    {
        public readonly float CenterX;
        public readonly float CenterY;
        public readonly float Width;
        public readonly float Height;

        public PoseAnchor(float cx, float cy, float w, float h)
        {
            CenterX = cx;
            CenterY = cy;
            Width = w;
            Height = h;
        }
    }

    /// <summary>
    /// CSV TextAsset から anchor 配列をロードするヘルパ。
    /// </summary>
    internal static class PoseAnchors
    {
        /// <summary>
        /// CSV テキストをパースして anchor 配列を返す。
        /// </summary>
        /// <param name="csv">"cx,cy,w,h\n..." 形式のテキスト全体。</param>
        /// <returns>1 anchor = 1 要素の配列。順序は CSV 行順。</returns>
        public static PoseAnchor[] Load(string csv)
        {
            if (string.IsNullOrEmpty(csv))
                throw new ArgumentException("Anchors CSV is null or empty.", nameof(csv));

            // 改行コード混在に耐えるため Split('\n') 後に \r を除去。
            string[] lines = csv.Split('\n');

            // まず有効行数を数えて 1 回だけ確保。
            int count = 0;
            for (int i = 0; i < lines.Length; i++)
            {
                if (!IsBlank(lines[i])) count++;
            }

            var anchors = new PoseAnchor[count];
            int idx = 0;
            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i];
                if (IsBlank(line)) continue;

                // 末尾 CR 除去（Windows 行末対応）
                if (line[line.Length - 1] == '\r') line = line.Substring(0, line.Length - 1);

                string[] cols = line.Split(',');
                if (cols.Length < 4)
                    throw new FormatException($"Anchor line {i} has only {cols.Length} columns; expected >=4.");

                float cx = float.Parse(cols[0], System.Globalization.CultureInfo.InvariantCulture);
                float cy = float.Parse(cols[1], System.Globalization.CultureInfo.InvariantCulture);
                float w  = float.Parse(cols[2], System.Globalization.CultureInfo.InvariantCulture);
                float h  = float.Parse(cols[3], System.Globalization.CultureInfo.InvariantCulture);

                anchors[idx++] = new PoseAnchor(cx, cy, w, h);
            }

            return anchors;
        }

        static bool IsBlank(string s)
        {
            if (string.IsNullOrEmpty(s)) return true;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c != ' ' && c != '\t' && c != '\r' && c != '\n') return false;
            }
            return true;
        }
    }
}
