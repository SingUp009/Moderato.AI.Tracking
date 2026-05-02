// SPDX-License-Identifier: MIT
// Assets/Scenes/Sandbox/TrackingServiceDemo.cs
//
// TrackingService の動作確認用サンドボックス MonoBehaviour（M8）。
// - ポーズ（33 点）/ 手（21×2 点）/ 顔（468 点）を 1 フレームで同時推論する。
// - Webcam を背景に表示し、3 モダリティのスケルトンを GL.LINES でオーバーレイする。

using System;
using System.Threading;
using Moderato.AI.Tracking;
using Moderato.AI.Tracking.Processor;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// TrackingService（3 モダリティ統合）の動作確認。
/// Inspector で 9 つのモデルアセットを設定してから Play する。
/// </summary>
[AddComponentMenu("Moderato/Sandbox/Tracking Service Demo")]
public class TrackingServiceDemo : MonoBehaviour
{
    [Header("Pose Models")]
    [SerializeField] ModelAsset m_PoseDetector;       // pose_detector.onnx
    [SerializeField] ModelAsset m_PoseLandmarker;     // pose_landmark.onnx
    [SerializeField] TextAsset  m_PoseAnchorsCsv;     // pose_detection_anchors.csv

    [Header("Hand Models")]
    [SerializeField] ModelAsset m_PalmDetector;       // hand_detector.onnx
    [SerializeField] ModelAsset m_HandLandmarker;     // hand_landmarks_detector.onnx
    [SerializeField] TextAsset  m_PalmAnchorsCsv;     // palm_detection_anchors.csv

    [Header("Face Models")]
    [SerializeField] ModelAsset m_FaceDetector;       // blaze_face_short_range.onnx
    [SerializeField] ModelAsset m_FaceLandmarker;     // face_landmark.onnx
    [SerializeField] TextAsset  m_FaceAnchorsCsv;     // face_detection_anchors.csv

    [Header("Webcam")]
    [SerializeField] int m_WebcamWidth  = 1280;
    [SerializeField] int m_WebcamHeight = 720;
    [SerializeField] int m_WebcamFps    = 30;

    [Header("Display")]
    [SerializeField] bool m_ShowDebug = true;

    WebCamTexture    m_WebcamTex;
    RenderTexture    m_Frame;
    TrackingService  m_Service;
    TrackingFrame    m_LastResult;
    Material         m_LineMaterial;
    CancellationTokenSource m_Cts;

    // ---- Pose スケルトン接続 ----
    static readonly (int a, int b)[] k_PoseConnections =
    {
        // 頭
        (0, 1),(1, 2),(2, 3),(3, 7),
        (0, 4),(4, 5),(5, 6),(6, 8),
        (9,10),
        // 胴体
        (11,12),(11,23),(12,24),(23,24),
        // 左腕
        (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
        // 右腕
        (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
        // 左脚
        (23,25),(25,27),(27,29),(29,31),(27,31),
        // 右脚
        (24,26),(26,28),(28,30),(30,32),(28,32),
    };

    // ---- Hand スケルトン接続 ----
    static readonly (int a, int b)[] k_HandConnections =
    {
        (0, 1),(1, 2),(2, 3),(3, 4),
        (0, 5),(5, 6),(6, 7),(7, 8),
        (0, 9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5, 9),(9,13),(13,17),
    };

    // ---- Face 輪郭インデックス ----
    static readonly int[] k_FaceSilhouette =
    {
        10,338,297,332,284,251,389,356,454,323,361,288,
        397,365,379,378,400,377,152,148,176,149,150,136,
        172,58,132,93,234,127,162,21,54,103,67,109,10,
    };
    static readonly int[] k_LeftEye  = { 33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33 };
    static readonly int[] k_RightEye = { 362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362 };
    static readonly int[] k_LipsOuter =
        { 61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61 };

    static readonly Color[] k_HandColors = { Color.green, Color.cyan };

    void Start()
    {
        if (!ValidateAssets()) return;

        m_Cts = new CancellationTokenSource();

        m_WebcamTex = new WebCamTexture(m_WebcamWidth, m_WebcamHeight, m_WebcamFps);
        m_WebcamTex.Play();

        m_Frame = new RenderTexture(m_WebcamWidth, m_WebcamHeight, 0, RenderTextureFormat.ARGB32)
        {
            name       = "TrackingServiceDemo.Frame",
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
            Debug.LogWarning("[TrackingServiceDemo] Hidden/Internal-Colored が見つかりません。ライン描画は無効です。");
        }

        var backend = SystemInfo.supportsComputeShaders
            ? BackendType.GPUCompute
            : BackendType.GPUPixel;

        m_Service = new TrackingService(
            m_PoseDetector,  m_PoseLandmarker, m_PoseAnchorsCsv,
            m_PalmDetector,  m_HandLandmarker, m_PalmAnchorsCsv,
            m_FaceDetector,  m_FaceLandmarker, m_FaceAnchorsCsv,
            backend);

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
                    m_LastResult = await m_Service.DetectAsync(m_Frame, ct);
                }
                await Awaitable.NextFrameAsync(ct);
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception e) { Debug.LogException(e); }
    }

    void OnGUI()
    {
        if (m_WebcamTex != null && m_WebcamTex.isPlaying)
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, Screen.height),
                m_WebcamTex, ScaleMode.StretchToFill, false);

        DrawHandRoiPreview();

        if (m_ShowDebug)
        {
            GUI.color = Color.white;
            GUI.Label(new Rect(10f, 10f, 600f, 22f),
                $"Pose: {(m_LastResult.HasPose ? "YES" : "no ")} | " +
                $"Hands: {(m_LastResult.Hands != null ? (m_LastResult.Hands[0].IsValid ? "L" : "_") + (m_LastResult.Hands[1].IsValid ? "R" : "_") : "--")} | " +
                $"Face: {(m_LastResult.HasFace ? "YES" : "no ")}");
        }

        // ---- Pose ドット（可視性 >= 0.5 のみ）----
        // Y は top-origin（0=上端）。OnGUI も top-origin なのでそのまま Y * Height。
        if (m_LastResult.HasPose)
        {
            GUI.color = new Color(1f, 0.6f, 0f, 0.8f);
            var lm = m_LastResult.Pose.Landmarks;
            for (int i = 0; i < lm.Length; i++)
            {
                if (lm[i].Visibility < 0.5f) continue;
                float px = lm[i].X * Screen.width;
                float py = lm[i].Y * Screen.height;
                GUI.Box(new Rect(px - 3f, py - 3f, 6f, 6f), string.Empty);
            }
        }

        // ---- Hand ドット（Y は top-origin → OnGUI はそのまま Y * Height）----
        if (m_LastResult.Hands != null)
        {
            for (int h = 0; h < m_LastResult.Hands.Length; h++)
            {
                var hf = m_LastResult.Hands[h];
                bool hasSignal = hf.IsValid || hf.DetectionScore > 0f || hf.PresenceScore > 0f;
                if (!hasSignal) continue;

                GUI.color = hf.IsValid ? k_HandColors[h % k_HandColors.Length] : new Color(1f, 0.35f, 0.35f, 1f);
                if (m_ShowDebug)
                {
                    GUI.Label(new Rect(10f, 34f + h * 20f, 520f, 20f),
                        $"Hand {h} | {(hf.IsValid ? hf.Hand.ToString() : "invalid")} | det={hf.DetectionScore:F2} | pres={hf.PresenceScore:F2}");
                }

                if (!hf.IsValid) continue;

                var lm = hf.Landmarks;
                for (int i = 0; i < lm.Length; i++)
                {
                    float px = lm[i].X * Screen.width;
                    float py = lm[i].Y * Screen.height;
                    GUI.Box(new Rect(px - 4f, py - 4f, 8f, 8f), string.Empty);
                }
            }
        }

        // ---- Face ドット（顔：Y は bottom-origin → 1f-Y で上下反転） ----
        if (m_LastResult.HasFace)
        {
            GUI.color = new Color(0f, 0.8f, 1f, 0.6f);
            var lm = m_LastResult.Face.Landmarks;
            for (int i = 0; i < lm.Length; i++)
            {
                float px = lm[i].X * Screen.width;
                float py = (1f - lm[i].Y) * Screen.height;
                GUI.Box(new Rect(px - 2f, py - 2f, 4f, 4f), string.Empty);
            }
        }

        GUI.color = Color.white;

        // ---- GL スケルトンライン ----
        if (Event.current.type != EventType.Repaint || m_LineMaterial == null) return;
        GL.PushMatrix();
        m_LineMaterial.SetPass(0);
        GL.LoadOrtho();
        GL.Begin(GL.LINES);

        // Pose: X ミラー補正(1f-X)、Y は top-origin なので 1f-Y で GL 反転。
        // 可視性 < 0.5 の接続（画外の足・腰）はスキップして画面全体への誤描画を防ぐ。
        if (m_LastResult.HasPose)
        {
            GL.Color(new Color(1f, 0.6f, 0f, 0.9f));
            var lm = m_LastResult.Pose.Landmarks;
            for (int ci = 0; ci < k_PoseConnections.Length; ci++)
            {
                int ia = k_PoseConnections[ci].a;
                int ib = k_PoseConnections[ci].b;
                if (lm[ia].Visibility < 0.5f || lm[ib].Visibility < 0.5f) continue;
                GL.Vertex3(1f - lm[ia].X, 1f - lm[ia].Y, 0f);
                GL.Vertex3(1f - lm[ib].X, 1f - lm[ib].Y, 0f);
            }
        }

        // Hand: X は OnGUI のドットと同じ、Y は top-origin なので 1f-Y で GL 反転
        if (m_LastResult.Hands != null)
        {
            for (int h = 0; h < m_LastResult.Hands.Length; h++)
            {
                var hf = m_LastResult.Hands[h];
                if (!hf.IsValid) continue;
                GL.Color(k_HandColors[h % k_HandColors.Length]);
                var lm = hf.Landmarks;
                for (int ci = 0; ci < k_HandConnections.Length; ci++)
                {
                    int ia = k_HandConnections[ci].a;
                    int ib = k_HandConnections[ci].b;
                    GL.Vertex3(lm[ia].X, 1f - lm[ia].Y, 0f);
                    GL.Vertex3(lm[ib].X, 1f - lm[ib].Y, 0f);
                }
            }
        }

        // Face: X 補正なし、Y は bottom-origin で GL と一致
        if (m_LastResult.HasFace)
        {
            var lm = m_LastResult.Face.Landmarks;
            DrawFacePolyline(lm, k_FaceSilhouette, new Color(0f,  0.8f, 1f,  0.9f));
            DrawFacePolyline(lm, k_LeftEye,        new Color(0.3f,0.8f, 1f,  0.9f));
            DrawFacePolyline(lm, k_RightEye,       new Color(0.3f,0.8f, 1f,  0.9f));
            DrawFacePolyline(lm, k_LipsOuter,      new Color(1f,  0.3f, 0.4f,0.9f));
        }

        GL.End();
        GL.PopMatrix();
    }

    void DrawHandRoiPreview()
    {
        Texture roi = m_Service != null ? m_Service.DebugHandRoiTexture : null;
        if (roi == null) return;

        const float size = 160f;
        var rect = new Rect(Screen.width - size - 10f, 10f, size, size);
        GUI.color = Color.white;
        GUI.DrawTexture(rect, roi, ScaleMode.StretchToFill, false);
        GUI.Box(rect, "Hand ROI");
    }

    static void DrawFacePolyline(FaceKeypoint[] lm, int[] indices, Color color)
    {
        GL.Color(color);
        for (int i = 0; i + 1 < indices.Length; i++)
        {
            var a = lm[indices[i]];
            var b = lm[indices[i + 1]];
            GL.Vertex3(a.X, a.Y, 0f);
            GL.Vertex3(b.X, b.Y, 0f);
        }
    }

    bool ValidateAssets()
    {
        if (m_PoseDetector   == null || m_PoseLandmarker  == null || m_PoseAnchorsCsv == null ||
            m_PalmDetector   == null || m_HandLandmarker  == null || m_PalmAnchorsCsv == null ||
            m_FaceDetector   == null || m_FaceLandmarker  == null || m_FaceAnchorsCsv == null)
        {
            Debug.LogError("[TrackingServiceDemo] Inspector で全モデルアセットを割り当ててください。");
            enabled = false;
            return false;
        }
        return true;
    }

    void OnDestroy()
    {
        m_Cts?.Cancel();
        m_Cts?.Dispose();
        m_Service?.Dispose();

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
