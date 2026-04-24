# Moderato.AI.Tracking — Claude Code ガイド

## プロジェクトの目的

Web カメラ映像から人間の **ポーズ（BlazePose, 33 点）/ 手（Hand Landmark, 21 点 × 両手）/ 顔（FaceMesh / Blendshape）** をリアルタイムに推定し、Unity シーン内のアバター・演出に反映する。

## 実行環境

- Unity Editor：`6000.4.4f1`（`ProjectSettings/ProjectVersion.txt` を参照）
- 対象プラットフォーム：**Windows / macOS スタンドアロン**
- 推論：**Unity Sentis（`com.unity.ai.inference 2.6.1`）** / バックエンドは `BackendType.GPUCompute`
- Barracuda は使用しない（Sentis に一本化）

## コーディング方針（厳守）

1. **非同期は `ValueTask` を使う。`IEnumerator` / Coroutine は原則禁止。**
   - 理由：コルーチンのステートマシン & `yield` はホットパスで GC Alloc の温床になるため。
   - `async ValueTask` を用い、`AsyncValueTaskMethodBuilder` のゼロアロケーション特性を活かす。必要なら UniTask の導入も検討。
2. **GC Alloc をホットパスで出さない（0 alloc 目標）。**
   - `Tensor` / `ComputeBuffer` / ランドマーク配列は起動時に確保して使い回す。
   - `$"..."` 文字列補間・`List<T>` の動的増加・boxing を伴う `foreach` は `Update()` 経路に置かない。
   - 開発時は Profiler の `GC Alloc` 列を常時 0 に保つ。
3. **推論は非同期 readback 前提**：`Worker.Schedule()` → 非同期 readback を `ValueTask` で包む。`PeekOutput()` の同期 readback は禁止。
4. **GPU テクスチャ直入力**：`TextureConverter.ToTensor(RenderTexture, ...)` を使い、CPU 経由のコピーを作らない。

## ディレクトリ構成（[`Moderato.Mathematics.FFT`](https://github.com/SingUp009/Moderato.Mathematics.FFT) 規約に準拠）

- **パッケージルート：`Assets/Moderato/`**（`package.json` はここ）
- Git URL インポート：`?path=Assets/Moderato`
- **コアロジックは必ず `Assets/Moderato/AI/Tracking/` 配下に置く。** `Assets/Scenes/` や `Assets/Scripts/Tests/` などパッケージ外には依存しない。
- 階層はフラット：`Runtime/` / `Editor/` ラッパは作らない。カテゴリ直下に `.asmdef` を置く。
- 公開 API は `AI/Tracking/Processor/`、内部実装は `AI/Tracking/Core/`（`internal` で閉じる）。
- 同梱 `.sentis` モデルは `AI/Tracking/Models/`（Git LFS 管理）。
- 動作確認用 Scene / 実験コードは `Assets/Scenes/Sandbox/` などパッケージ外に置く。
- テストは `Assets/Scripts/Tests/` にパッケージ外で配置し、`UNITY_INCLUDE_TESTS` define で切り替え。

## バージョン管理

- **Git + Git LFS**（`.onnx` / `.sentis` / 画像 / 動画 / FBX / wav は LFS）
- `.unity` / `.prefab` / `.asset` の merge driver は **UnityYAMLMerge**
- **各作業単位で都度コミットする。** マイルストーン（モデル単体完成・パイプライン統合など）ごとに必ずコミットを残す。
- コミットメッセージ prefix：`feat:` / `fix:` / `chore:` / `refactor:` / `perf:` / `test:`

## パッケージ配布（Git URL インポート）

- `Assets/Moderato/` 全体を UPM パッケージとして配布：

  ```
  https://github.com/<owner>/Moderato.AI.Tracking.git?path=Assets/Moderato
  ```

- **パッケージ境界を厳守**：コアロジックは `Assets/Moderato/AI/Tracking/` の内側で完結させる。`using` / `Resources.Load` / `AssetDatabase.FindAssets` などあらゆる依存経路でパッケージ外（`Assets/Scripts/`, `Assets/Scenes/`, `Assets/Resources/`）を参照しないこと。Git URL 経由で消費側が動かなくなる。

## 検収条件

新機能を `main` にマージする前に、以下を満たすこと：

- Profiler の `GC Alloc / frame` がゼロ
- 目標 30 fps が Windows / macOS で維持できている
- 3 モダリティ（ポーズ・手・顔）が同時に動作している
