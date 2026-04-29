// SPDX-License-Identifier: MIT
// Assets/Scenes/Sandbox/HandTrackingDemo.cs
//
// HandLandmarker の動作確認用サンドボックス MonoBehaviour。
// - WebCamTexture を StretchToFill で表示し、ランドマーク座標と 1:1 で一致させる。
// - GUI.Box で 21 点のドット、GL.LINES でスケルトン接続を描画する。

using System;
using System.Threading;
using Moderato.AI.Tracking;
using Moderato.AI.Tracking.Processor;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// HandLandmarker の動作確認。
/// Inspector で 3 つのアセットを設定してから Play する。
/// </summary>
[AddComponentMenu("Moderato/Sandbox/Hand Tracking Demo")]
public class HandTrackingDemo : MonoBehaviour
{
    [Header("Models — Inspector で割り当てること")]
    [SerializeField] ModelAsset m_PalmDetector;     // hand_detector.onnx
    [SerializeField] ModelAsset m_HandLandmarker;   // hand_landmarks_detector.onnx
    [SerializeField] TextAsset  m_PalmAnchorsCsv;   // palm_detection_anchors.csv

    [Header("Webcam")]
    [SerializeField] int m_WebcamWidth  = 1280;
    [SerializeField] int m_WebcamHeight = 720;
    [SerializeField] int m_WebcamFps    = 30;

    WebCamTexture  m_WebcamTex;
    RenderTexture  m_Frame;
    HandLandmarker m_Landmarker;
    HandFrame[]    m_LastResult;
    Material       m_LineMaterial;
    CancellationTokenSource m_Cts;

    // 1手目=緑、2手目=シアン
    static readonly Color[] k_HandColors = { Color.green, Color.cyan };

    // BlazeHand スケルトン接続定義（MediaPipe 標準 21 点）
    static readonly (int a, int b)[] k_Connections =
    {
        (0, 1),(1, 2),(2, 3),(3, 4),           // 親指
        (0, 5),(5, 6),(6, 7),(7, 8),           // 人差し指
        (0, 9),(9,10),(10,11),(11,12),          // 中指
        (0,13),(13,14),(14,15),(15,16),         // 薬指
        (0,17),(17,18),(18,19),(19,20),         // 小指
        (5, 9),(9,13),(13,17),                  // 手のひら横断
    };

    void Start()
    {
        if (m_PalmDetector == null || m_HandLandmarker == null || m_PalmAnchorsCsv == null)
        {
            Debug.LogError("[HandTrackingDemo] Inspector でモデルアセットを全て割り当ててください。");
            enabled = false;
            return;
        }

        m_Cts = new CancellationTokenSource();

        // Webcam → RenderTexture パイプライン
        m_WebcamTex = new WebCamTexture(m_WebcamWidth, m_WebcamHeight, m_WebcamFps);
        m_WebcamTex.Play();

        m_Frame = new RenderTexture(m_WebcamWidth, m_WebcamHeight, 0, RenderTextureFormat.ARGB32)
        {
            name       = "HandTrackingDemo.Frame",
            wrapMode   = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
        };
        m_Frame.Create();

        // GL ライン描画用マテリアル（起動時 1 回だけ確保）
        var shader = Shader.Find("Hidden/Internal-Colored");
        if (shader != null)
        {
            m_LineMaterial = new Material(shader) { hideFlags = HideFlags.HideAndDontSave };
            m_LineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            m_LineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            m_LineMaterial.SetInt("_Cull",     (int)UnityEngine.Rendering.CullMode.Off);
            m_LineMaterial.SetInt("_ZWrite",   0);
        }
        else
        {
            Debug.LogWarning("[HandTrackingDemo] Hidden/Internal-Colored が見つかりません。ライン描画は無効です。");
        }

        // GPU バックエンドを選択（SentisBackendFactory は internal のためインライン）
        var backend = SystemInfo.supportsComputeShaders
            ? BackendType.GPUCompute
            : BackendType.GPUPixel;

        m_Landmarker = new HandLandmarker(
            m_PalmDetector, m_HandLandmarker, m_PalmAnchorsCsv, backend);

        RunLoopAsync(m_Cts.Token);
    }

    async void RunLoopAsync(CancellationToken ct)
    {
        try
        {
            while (!ct.IsCancellationRequested)
            {
                if (m_WebcamTex.isPlaying && m_WebcamTex.didUpdateThisFrame)
                {
                    Graphics.Blit(m_WebcamTex, m_Frame);
                    m_LastResult = await m_Landmarker.DetectAsync(m_Frame, ct);
                }

                await Awaitable.NextFrameAsync(ct);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e) { Debug.LogException(e); }
    }

    void OnGUI()
    {
        // ---- Webcam 映像（StretchToFill でランドマーク座標と完全一致） ----
        if (m_WebcamTex != null && m_WebcamTex.isPlaying)
        {
            GUI.DrawTexture(
                new Rect(0f, 0f, Screen.width, Screen.height),
                m_WebcamTex, ScaleMode.StretchToFill, false);
        }

        // ---- ステータスヘッダ ----
        GUI.color = Color.white;
        GUI.Label(new Rect(10f, 10f, 420f, 24f),
            $"Hands detected: {m_Landmarker?.DetectedHandCount ?? 0}");

        if (m_LastResult == null) return;

        // ---- ドット & ラベル ----
        for (int h = 0; h < m_LastResult.Length; h++)
        {
            var frame = m_LastResult[h];
            if (!frame.IsValid) continue;

            GUI.color = k_HandColors[h % k_HandColors.Length];
            GUI.Label(new Rect(10f, 38f + h * 22f, 420f, 22f),
                $"Hand {h} | {frame.Hand} | det={frame.DetectionScore:F2} | pres={frame.PresenceScore:F2}");

            var lm = frame.Landmarks;
            for (int i = 0; i < lm.Length; i++)
            {
                float px = lm[i].X * Screen.width;
                // Y は bottom-origin。OnGUI は top-origin なので (1f - Y) で変換。
                float py = (1f - lm[i].Y) * Screen.height;
                GUI.Box(new Rect(px - 5f, py - 5f, 10f, 10f), string.Empty);
            }
        }

        GUI.color = Color.white;

        // ---- GL スケルトンライン（Repaint イベントのみ）----
        // GL.LoadOrtho() は (0,0)=左下 (1,1)=右上。
        // ランドマーク X は 1f - X でミラー補正。Y は GL Y=0=下と相殺されるためそのまま。
        if (Event.current.type != EventType.Repaint || m_LineMaterial == null) return;

        GL.PushMatrix();
        m_LineMaterial.SetPass(0);
        GL.LoadOrtho();
        GL.Begin(GL.LINES);

        for (int h = 0; h < m_LastResult.Length; h++)
        {
            var frame = m_LastResult[h];
            if (!frame.IsValid) continue;

            GL.Color(k_HandColors[h % k_HandColors.Length]);
            var lm = frame.Landmarks;

            for (int ci = 0; ci < k_Connections.Length; ci++)
            {
                int ia = k_Connections[ci].a;
                int ib = k_Connections[ci].b;
                GL.Vertex3(1f - lm[ia].X, lm[ia].Y, 0f);
                GL.Vertex3(1f - lm[ib].X, lm[ib].Y, 0f);
            }
        }

        GL.End();
        GL.PopMatrix();
    }

    void OnDestroy()
    {
        m_Cts?.Cancel();
        m_Cts?.Dispose();

        m_Landmarker?.Dispose();

        if (m_LineMaterial != null)
            Destroy(m_LineMaterial);

        if (m_Frame != null)
        {
            m_Frame.Release();
            Destroy(m_Frame);
        }

        if (m_WebcamTex != null)
        {
            m_WebcamTex.Stop();
            Destroy(m_WebcamTex);
        }
    }
}
