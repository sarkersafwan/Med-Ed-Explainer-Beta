export type SceneMode = "avatar" | "animation" | "overlay";

export interface SceneData {
  sceneNumber: number;
  title: string;
  mode: SceneMode;
  durationFrames: number;
  voiceFile: string | null;
  avatarFile: string | null;
  segments: SegmentData[];
  textOverlays: TextOverlay[];
}

export interface SegmentData {
  segmentIndex: number;
  title: string;
  animationFile: string | null;
  imageFile: string | null;
  durationFrames: number;
  /** Actual length of the underlying mp4 in frames (0 if no animation file). */
  animationDurationFrames: number;
}

export type OverlayEmphasis = "clinical" | "label";

export interface TextOverlay {
  text: string;
  startFrame: number;
  durationFrames: number;
  emphasis: OverlayEmphasis;
}

export interface VideoProps {
  scenes: SceneData[];
  audioFile: string;
  fps: number;
  totalDurationFrames: number;
}
