# Moderato.AI.Tracking

Realtime **pose / hand / face** tracking in Unity using [Sentis](https://docs.unity3d.com/Packages/com.unity.sentis@latest) (ONNX).

## Features

- BlazePose (33 landmarks) で全身ポーズ推定
- Hand Landmark (21 landmarks × 両手) で手指追跡
- FaceMesh / Blendshape で表情推定
- 3 モダリティを Web カメラ入力から同時・リアルタイム（目標 30 fps, Windows / macOS）

## Requirements

- Unity **6000.4.4f1** 以降（Unity 6）
- `com.unity.ai.inference` 2.6.1（Sentis）
- **Git LFS**（同梱 `.sentis` モデル取得のため消費側マシンにも必須）

## Install via Git URL

Package Manager → `+` → `Add package from git URL...` から以下を追加：

```
https://github.com/<owner>/Moderato.AI.Tracking.git?path=Assets/Moderato
```

または `Packages/manifest.json` に追記：

```json
{
  "dependencies": {
    "com.singup_009.moderato.ai.tracking": "https://github.com/<owner>/Moderato.AI.Tracking.git?path=Assets/Moderato"
  }
}
```

バージョン固定は URL 末尾に `#v0.1.0`（タグ）や `#<commit-sha>` を付ける。

> **Note**: 消費側マシンに Git LFS が入っていないと、`.sentis` モデルファイルがポインタのまま取得され、ロードに失敗する。`git lfs install` を済ませてから Package Manager を使うこと。

## License

MIT — see [LICENSE](LICENSE).

## Related

- [Moderato.Mathematics.FFT](https://github.com/SingUp009/Moderato.Mathematics.FFT) — 同著者の別 Unity パッケージ（本リポジトリの規約リファレンス）
