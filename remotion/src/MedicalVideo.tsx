import { AbsoluteFill, Sequence, Series } from "remotion";
import { SceneClip } from "./SceneClip";
import type { VideoProps } from "./types";

/**
 * Scenes play back-to-back via Remotion's Series (no overlap, no transition
 * frames eaten). Each scene owns its own audio so the voice track is always
 * frame-locked to the avatar video — no global audio track to drift against.
 */
export const MedicalVideo: React.FC<VideoProps> = ({ scenes }) => {
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <Series>
        {scenes.map((scene, i) => (
          <Series.Sequence
            key={`scene-${i}`}
            durationInFrames={scene.durationFrames}
          >
            <SceneClip scene={scene} />
          </Series.Sequence>
        ))}
      </Series>
    </AbsoluteFill>
  );
};
