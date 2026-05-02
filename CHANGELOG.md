# Changelog

## 0.2.2 - 2026-05-02

### Added
- Added a rotated ROI blit shader for BlazeHand and wired it into `HandLandmarker` with an axis-aligned fallback.
- Added hand ROI debug preview surfaces to the hand sandbox demo and the integrated tracking service demo.
- Added the `FaceTrackingDemo` scene asset.
- Added Unity MCP project support and Roslyn editor assemblies for local Unity automation.

### Changed
- Reworked Hand ROI generation to follow the Unity BlazeHand sample flow: palm box size, wrist to middle-MCP rotation, shifted ROI center, and 2.6x ROI scaling.
- Updated hand landmark projection so rotated crops and inverse projection use the same top-origin coordinate model.
- Stopped applying a second sigmoid to BlazeHand presence output and kept low-confidence palm candidates for weaker hand poses.
- Sorted valid hand frames by screen X so two-hand output order is stable.
- Lowered the face detector threshold from `0.75` to `0.5`.
- Updated sandbox drawing rules for confirmed coordinate systems: Face no longer mirrors X, and Hand GL output flips only Y.
- Updated `AGENTS.md` and `CLAUDE.md` with the current coordinate-system and Hand ROI findings.

### Fixed
- Fixed the Hand landmark 90-degree drift caused by mismatched ROI rotation between crop and projection.
- Fixed stale or misplaced Hand landmarks being drawn after weak or invalid landmark presence results.
