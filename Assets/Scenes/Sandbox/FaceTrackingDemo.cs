// SPDX-License-Identifier: MIT
// Assets/Scenes/Sandbox/FaceTrackingDemo.cs
//
// FaceLandmarker の動作確認用サンドボックス MonoBehaviour。
// - WebCamTexture を StretchToFill で表示し、ランドマーク座標と 1:1 で一致させる。
// - GUI.Box で 468 点のドット、GL.LINES で顔輪郭・眼・眉・鼻・口のラインを描画する。

using System;
using System.Threading;
using Moderato.AI.Tracking;
using Moderato.AI.Tracking.Processor;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// FaceLandmarker の動作確認。
/// Inspector で 3 つのアセットを設定してから Play する。
/// </summary>
[AddComponentMenu("Moderato/Sandbox/Face Tracking Demo")]
public class FaceTrackingDemo : MonoBehaviour
{
    [Header("Models — Inspector で割り当てること")]
    [SerializeField] ModelAsset m_FaceDetector;    // blaze_face_short_range.onnx
    [SerializeField] ModelAsset m_FaceLandmarker;  // face_landmark.onnx
    [SerializeField] TextAsset  m_FaceAnchorsCsv;  // face_detection_anchors.csv

    [Header("Webcam")]
    [SerializeField] int m_WebcamWidth  = 1280;
    [SerializeField] int m_WebcamHeight = 720;
    [SerializeField] int m_WebcamFps    = 30;

    [Header("Display")]
    [SerializeField] bool m_ShowDebug = true;

    WebCamTexture   m_WebcamTex;
    RenderTexture   m_Frame;
    FaceLandmarker  m_Landmarker;
    FaceFrame       m_LastResult;
    Material        m_LineMaterial;
    CancellationTokenSource m_Cts;

    // ---- 描画ライン定義（MediaPipe FaceMesh 標準インデックス） ----

    // 顔輪郭（silhouette）
    static readonly int[] k_Silhouette =
    {
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
    };

    // 左眼
    static readonly int[] k_LeftEye =
        { 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33 };

    // 右眼
    static readonly int[] k_RightEye =
        { 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362 };

    // 左眉毛
    static readonly int[] k_LeftEyebrow =
        { 46, 53, 52, 65, 55, 70, 63, 105, 66, 107 };

    // 右眉毛
    static readonly int[] k_RightEyebrow =
        { 276, 283, 282, 295, 285, 300, 293, 334, 296, 336 };

    // 鼻筋
    static readonly int[] k_Nose =
        { 168, 6, 197, 195, 5, 4, 1, 19, 94, 2 };

    // 口（外側）
    static readonly int[] k_LipsOuter =
        { 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185, 61 };

    // 口（内側）
    static readonly int[] k_LipsInner =
        { 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78 };

    void Start()
    {
        if (m_FaceDetector == null || m_FaceLandmarker == null || m_FaceAnchorsCsv == null)
        {
            Debug.LogError("[FaceTrackingDemo] Inspector でモデルアセットを全て割り当ててください。");
            enabled = false;
            return;
        }

        m_Cts = new CancellationTokenSource();

        m_WebcamTex = new WebCamTexture(m_WebcamWidth, m_WebcamHeight, m_WebcamFps);
        m_WebcamTex.Play();

        m_Frame = new RenderTexture(m_WebcamWidth, m_WebcamHeight, 0, RenderTextureFormat.ARGB32)
        {
            name       = "FaceTrackingDemo.Frame",
            wrapMode   = TextureWrapMode.Clamp,
            filterMode = FilterMode.Bilinear,
        };
        m_Frame.Create();

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
            Debug.LogWarning("[FaceTrackingDemo] Hidden/Internal-Colored が見つかりません。ライン描画は無効です。");
        }

        var backend = SystemInfo.supportsComputeShaders
            ? BackendType.GPUCompute
            : BackendType.GPUPixel;

        m_Landmarker = new FaceLandmarker(
            m_FaceDetector, m_FaceLandmarker, m_FaceAnchorsCsv, backend);

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
            GUI.DrawTexture(
                new Rect(0f, 0f, Screen.width, Screen.height),
                m_WebcamTex, ScaleMode.StretchToFill, false);

        // ---- ステータス ----
        if (m_ShowDebug)
        {
            GUI.color = Color.white;
            GUI.Label(new Rect(10f, 10f, 520f, 24f),
                $"Face detected: {(m_LastResult.IsValid ? "YES" : "NO")} | " +
                $"det={m_Landmarker?.LastDetectionScore:F2} | " +
                $"pres={m_LastResult.PresenceScore:F2}");
        }

        if (!m_LastResult.IsValid) return;

        var lm = m_LastResult.Landmarks;

        // ---- 468 点ドット ----
        GUI.color = new Color(0f, 1f, 0.5f, 0.7f);
        for (int i = 0; i < lm.Length; i++)
        {
            // WebCamTexture は水平ミラー表示なので X を反転
            float px = (1f - lm[i].X) * Screen.width;
            float py = (1f - lm[i].Y) * Screen.height;
            GUI.Box(new Rect(px - 2f, py - 2f, 4f, 4f), string.Empty);
        }
        GUI.color = Color.white;

        // ---- GL スケルトンライン（Repaint イベントのみ）----
        // GL.LoadOrtho() は (0,0)=左下 (1,1)=右上。
        // ランドマーク Y=0 が上 / GL Y=0 が下 → Y はそのまま渡すと相殺。
        // ランドマーク X は水平ミラー補正で (1 - X)。
        if (Event.current.type != EventType.Repaint || m_LineMaterial == null) return;

        GL.PushMatrix();
        m_LineMaterial.SetPass(0);
        GL.LoadOrtho();
        GL.Begin(GL.LINES);

        DrawPolyline(lm, k_Silhouette,    new Color(0.0f, 1.0f, 0.5f, 0.9f));
        DrawPolyline(lm, k_LeftEye,       new Color(0.3f, 0.8f, 1.0f, 0.9f));
        DrawPolyline(lm, k_RightEye,      new Color(0.3f, 0.8f, 1.0f, 0.9f));
        DrawPolyline(lm, k_LeftEyebrow,   new Color(1.0f, 0.8f, 0.2f, 0.9f));
        DrawPolyline(lm, k_RightEyebrow,  new Color(1.0f, 0.8f, 0.2f, 0.9f));
        DrawPolyline(lm, k_Nose,          new Color(1.0f, 0.5f, 0.2f, 0.9f));
        DrawPolyline(lm, k_LipsOuter,     new Color(1.0f, 0.3f, 0.4f, 0.9f));
        DrawPolyline(lm, k_LipsInner,     new Color(1.0f, 0.3f, 0.4f, 0.7f));

        GL.End();
        GL.PopMatrix();
    }

    static void DrawPolyline(FaceKeypoint[] lm, int[] indices, Color color)
    {
        GL.Color(color);
        for (int i = 0; i + 1 < indices.Length; i++)
        {
            var a = lm[indices[i]];
            var b = lm[indices[i + 1]];
            GL.Vertex3(1f - a.X, a.Y, 0f);
            GL.Vertex3(1f - b.X, b.Y, 0f);
        }
    }

    void OnDestroy()
    {
        m_Cts?.Cancel();
        m_Cts?.Dispose();
        m_Landmarker?.Dispose();

        if (m_LineMaterial != null) Destroy(m_LineMaterial);

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
