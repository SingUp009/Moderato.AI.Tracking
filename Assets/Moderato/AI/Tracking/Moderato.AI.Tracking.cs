using UnityEngine;

namespace Moderato.AI.Tracking
{
    public static class TrackingConstants
    {
        /// <summary>BlazePose のランドマーク数</summary>
        public const int PoseLandmarkCount = 33;

        /// <summary>Hand Landmark のランドマーク数（片手）</summary>
        public const int HandLandmarkCount = 21;

        /// <summary>FaceMesh のランドマーク数</summary>
        public const int FaceLandmarkCount = 468;

        /// <summary>同時検出できる手の最大数</summary>
        public const int MaxHandCount = 2;
    }

    /// <summary>
    /// MediaPipe BlazePose 33 点のインデックス定義
    /// </summary>
    public enum PoseLandmark
    {
        Nose = 0,
        LeftEyeInner = 1,
        LeftEye = 2,
        LeftEyeOuter = 3,
        RightEyeInner = 4,
        RightEye = 5,
        RightEyeOuter = 6,
        LeftEar = 7,
        RightEar = 8,
        MouthLeft = 9,
        MouthRight = 10,
        LeftShoulder = 11,
        RightShoulder = 12,
        LeftElbow = 13,
        RightElbow = 14,
        LeftWrist = 15,
        RightWrist = 16,
        LeftPinky = 17,
        RightPinky = 18,
        LeftIndex = 19,
        RightIndex = 20,
        LeftThumb = 21,
        RightThumb = 22,
        LeftHip = 23,
        RightHip = 24,
        LeftKnee = 25,
        RightKnee = 26,
        LeftAnkle = 27,
        RightAnkle = 28,
        LeftHeel = 29,
        RightHeel = 30,
        LeftFootIndex = 31,
        RightFootIndex = 32,
    }

    /// <summary>
    /// 1 ランドマークの座標と信頼度
    /// </summary>
    public readonly struct PoseKeypoint
    {
        public readonly float X;
        public readonly float Y;
        public readonly float Z;
        public readonly float Visibility;
        public readonly float Presence;

        public PoseKeypoint(float x, float y, float z, float visibility, float presence)
        {
            X = x;
            Y = y;
            Z = z;
            Visibility = visibility;
            Presence = presence;
        }

        public Vector2 ToVector2() => new Vector2(X, Y);

        public Vector3 ToVector3() => new Vector3(X, Y, Z);
    }

    public readonly struct PoseFrame
    {
        public readonly PoseKeypoint[] Landmarks;

        public readonly float DetectionScore;

        public readonly bool IsValid;

        public PoseFrame(PoseKeypoint[] landmarks, float detectionScore, bool isValid)
        {
            Landmarks = landmarks;
            DetectionScore = detectionScore;
            IsValid = isValid;
        }
    }

    // -------------------------------------------------------------------------
    // Hand Landmark (M6)
    // -------------------------------------------------------------------------

    /// <summary>
    /// BlazeHand 21 点のインデックス定義（MediaPipe 標準番号）
    /// </summary>
    public enum HandLandmark
    {
        Wrist = 0,
        ThumbCmc = 1, ThumbMcp = 2, ThumbIp = 3, ThumbTip = 4,
        IndexMcp = 5, IndexPip = 6, IndexDip = 7, IndexTip = 8,
        MiddleMcp = 9, MiddlePip = 10, MiddleDip = 11, MiddleTip = 12,
        RingMcp = 13, RingPip = 14, RingDip = 15, RingTip = 16,
        PinkyMcp = 17, PinkyPip = 18, PinkyDip = 19, PinkyTip = 20,
    }

    /// <summary>
    /// 利き手識別。
    /// <para>
    /// MediaPipe の Handedness はカメラ画像基準であり、解剖学的な左右とは一致しない場合がある。
    /// 前面カメラ（鏡像）では Right が解剖学的左手に対応することに注意。
    /// </para>
    /// </summary>
    public enum Handedness : byte
    {
        Left    = 0,
        Right   = 1,
        Unknown = 255,
    }

    /// <summary>
    /// 1 ランドマークの座標（手モデルは点ごとの visibility / presence を持たない）
    /// </summary>
    public readonly struct HandKeypoint
    {
        public readonly float X;
        public readonly float Y;
        /// <summary>深度。224×224 クロップ画像のピクセル単位（正規化なし）。</summary>
        public readonly float Z;

        public HandKeypoint(float x, float y, float z)
        {
            X = x; Y = y; Z = z;
        }

        public Vector2 ToVector2() => new Vector2(X, Y);
        public Vector3 ToVector3() => new Vector3(X, Y, Z);
    }

    /// <summary>
    /// 1 フレーム・1 手分の検出結果。
    /// <para>
    /// <see cref="Landmarks"/> は内部バッファの参照。フレームをまたいでキャッシュしないこと。
    /// </para>
    /// </summary>
    public readonly struct HandFrame
    {
        /// <summary>21 点ランドマーク（入力画像正規化座標）。<see cref="IsValid"/> が false のとき内容は未定義。</summary>
        public readonly HandKeypoint[] Landmarks;

        /// <summary>利き手識別。</summary>
        public readonly Handedness Hand;

        /// <summary>Palm detector のシグモイドスコア。</summary>
        public readonly float DetectionScore;

        /// <summary>Hand landmark モデルの presence スコア（シグモイド適用済み）。</summary>
        public readonly float PresenceScore;

        /// <summary>有効な検出かどうか。</summary>
        public readonly bool IsValid;

        public HandFrame(HandKeypoint[] landmarks, Handedness hand,
                         float detectionScore, float presenceScore, bool isValid)
        {
            Landmarks      = landmarks;
            Hand           = hand;
            DetectionScore = detectionScore;
            PresenceScore  = presenceScore;
            IsValid        = isValid;
        }
    }

    // -------------------------------------------------------------------------
    // Face Landmark (M7)
    // -------------------------------------------------------------------------

    /// <summary>
    /// MediaPipe FaceMesh 468 点の主要ランドマーク名。
    /// <para>
    /// <see cref="LeftIrisCenter"/> と <see cref="RightIrisCenter"/> は
    /// with_attention モデル（478 点）でのみ有効。基本モデルでは範囲外になる。
    /// </para>
    /// </summary>
    public enum FaceLandmark
    {
        UpperLipTop       = 0,
        NoseTip           = 4,
        Forehead          = 10,
        LowerLipBottom    = 17,
        NoseBottom        = 94,
        Chin              = 152,
        LeftEyeOuter      = 33,
        LeftEyeInner      = 133,
        LeftEyeTop        = 159,
        LeftEyeBottom     = 145,
        LeftEyebrowOuter  = 46,
        LeftEyebrowInner  = 107,
        MouthLeft         = 78,
        UpperLipCenter    = 13,
        LowerLipCenter    = 14,
        RightEyeOuter     = 263,
        RightEyeInner     = 362,
        RightEyeTop       = 386,
        RightEyeBottom    = 374,
        RightEyebrowOuter = 276,
        RightEyebrowInner = 336,
        MouthRight        = 308,
        LeftIrisCenter    = 468,  // with_attention モデルのみ（478 点版）
        RightIrisCenter   = 473,  // with_attention モデルのみ（478 点版）
    }

    /// <summary>
    /// 顔ランドマーク 1 点の正規化座標。
    /// </summary>
    public readonly struct FaceKeypoint
    {
        public readonly float X;
        public readonly float Y;
        /// <summary>深度。ROI サイズ相対のピクセル単位（正規化なし）。</summary>
        public readonly float Z;

        public FaceKeypoint(float x, float y, float z)
        {
            X = x; Y = y; Z = z;
        }

        public Vector2 ToVector2() => new Vector2(X, Y);
        public Vector3 ToVector3() => new Vector3(X, Y, Z);
    }

    /// <summary>
    /// 1 フレーム分の顔検出結果（単一顔）。
    /// <para>
    /// <see cref="Landmarks"/> は内部バッファの参照。フレームをまたいでキャッシュしないこと。
    /// </para>
    /// </summary>
    public readonly struct FaceFrame
    {
        /// <summary>468 点ランドマーク（入力画像正規化座標）。<see cref="IsValid"/> が false のとき内容は未定義。</summary>
        public readonly FaceKeypoint[] Landmarks;

        /// <summary>BlazeFace detector のシグモイドスコア。</summary>
        public readonly float DetectionScore;

        /// <summary>Face landmark モデルの presence スコア（face_flag、シグモイド適用済み）。</summary>
        public readonly float PresenceScore;

        /// <summary>有効な検出かどうか。</summary>
        public readonly bool IsValid;

        public FaceFrame(FaceKeypoint[] landmarks, float detectionScore, float presenceScore, bool isValid)
        {
            Landmarks      = landmarks;
            DetectionScore = detectionScore;
            PresenceScore  = presenceScore;
            IsValid        = isValid;
        }
    }

    // -------------------------------------------------------------------------
    // Unified Tracking Frame (M8)
    // -------------------------------------------------------------------------

    /// <summary>
    /// 1 フレーム分の 3 モダリティ統合検出結果。
    /// <para>
    /// <see cref="Pose"/>, <see cref="Hands"/>, <see cref="Face"/> に含まれる配列は
    /// 各 Processor 内部バッファの参照。フレームをまたいでキャッシュしないこと。
    /// </para>
    /// </summary>
    public readonly struct TrackingFrame
    {
        /// <summary>ポーズ推定結果（33 点）。</summary>
        public readonly PoseFrame Pose;

        /// <summary>手ランドマーク結果（最大 2 手）。インデックス 0/1 が各手に対応する。</summary>
        public readonly HandFrame[] Hands;

        /// <summary>顔ランドマーク結果（468 点）。</summary>
        public readonly FaceFrame Face;

        public TrackingFrame(PoseFrame pose, HandFrame[] hands, FaceFrame face)
        {
            Pose  = pose;
            Hands = hands;
            Face  = face;
        }

        public bool HasPose    => Pose.IsValid;
        /// <summary>Hands は常に 2 要素配列（HandLandmarker 内部バッファ）。null チェック不要。</summary>
        public bool HasAnyHand => Hands[0].IsValid || Hands[1].IsValid;
        public bool HasFace    => Face.IsValid;
    }
}
