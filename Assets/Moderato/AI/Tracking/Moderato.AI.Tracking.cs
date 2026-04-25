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
}
