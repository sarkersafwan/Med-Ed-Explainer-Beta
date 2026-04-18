import {
  AbsoluteFill,
  Sequence,
  Img,
  OffthreadVideo,
  useVideoConfig,
  useCurrentFrame,
  interpolate,
  staticFile,
} from "remotion";
import { Audio } from "@remotion/media";
import { TextOverlay } from "./TextOverlay";
import type { SceneData, SegmentData } from "./types";

interface Props {
  scene: SceneData;
}

/**
 * Ken Burns animated image — slow zoom and slight pan
 * for cinematic visual interest on still frames.
 */
const KenBurnsImage: React.FC<{ src: string; durationFrames: number }> = ({
  src,
  durationFrames,
}) => {
  const frame = useCurrentFrame();
  const progress = frame / Math.max(durationFrames, 1);

  // Slow zoom from 1.0x → 1.15x
  const scale = interpolate(progress, [0, 1], [1.0, 1.15], {
    extrapolateRight: "clamp",
  });

  // Subtle horizontal pan: drift left-to-right
  const translateX = interpolate(progress, [0, 1], [-1.5, 1.5], {
    extrapolateRight: "clamp",
  });

  // Subtle vertical drift
  const translateY = interpolate(progress, [0, 1], [0.5, -0.5], {
    extrapolateRight: "clamp",
  });

  // Fade in at start, fade out at end for smooth transitions
  // Guard against short segments where keyframes would overlap
  const fadeDur = Math.min(8, Math.floor(durationFrames / 4));
  const opacity = fadeDur > 0
    ? interpolate(
        frame,
        [0, fadeDur, durationFrames - fadeDur, durationFrames],
        [0, 1, 1, 0],
        { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
      )
    : 1;

  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      <Img
        src={staticFile(src)}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          transform: `scale(${scale}) translate(${translateX}%, ${translateY}%)`,
          opacity,
        }}
      />
    </AbsoluteFill>
  );
};

/**
 * Render a single segment slot, looping the underlying animation clip if its
 * native duration is shorter than the slot it has to fill. Falls back to a
 * still image (Ken Burns) if no animation is available, and a dark plate if
 * neither asset exists.
 */
const SegmentSlot: React.FC<{ seg: SegmentData }> = ({ seg }) => {
  if (seg.animationFile) {
    const animFrames = seg.animationDurationFrames || seg.durationFrames;
    // How many full loops fit (at least 1) — render each as its own Sequence
    // so OffthreadVideo always plays from frame 0 instead of stalling at end.
    const loops = Math.max(1, Math.ceil(seg.durationFrames / animFrames));
    let cursor = 0;
    return (
      <AbsoluteFill style={{ backgroundColor: "#000" }}>
        {Array.from({ length: loops }).map((_, i) => {
          const from = cursor;
          const remaining = seg.durationFrames - cursor;
          const dur = Math.min(animFrames, remaining);
          cursor += dur;
          if (dur <= 0) return null;
          return (
            <Sequence key={`loop-${i}`} from={from} durationInFrames={dur}>
              <OffthreadVideo
                src={staticFile(seg.animationFile!)}
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
                muted
              />
            </Sequence>
          );
        })}
      </AbsoluteFill>
    );
  }

  if (seg.imageFile) {
    return (
      <KenBurnsImage src={seg.imageFile} durationFrames={seg.durationFrames} />
    );
  }

  return <AbsoluteFill style={{ backgroundColor: "#0a0a1a" }} />;
};

export const SceneClip: React.FC<Props> = ({ scene }) => {
  const { width, height } = useVideoConfig();

  // Avatar-mode scenes: full-screen talking head, no segments, no PIP.
  if (scene.mode === "avatar" && scene.avatarFile) {
    return (
      <AbsoluteFill style={{ backgroundColor: "#000" }}>
        <OffthreadVideo
          src={staticFile(scene.avatarFile)}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
          muted
        />
        {scene.voiceFile && <Audio src={staticFile(scene.voiceFile)} />}
        {scene.textOverlays.map((overlay, i) => (
          <Sequence
            key={`text-${i}`}
            from={overlay.startFrame}
            durationInFrames={overlay.durationFrames}
            premountFor={5}
          >
            <TextOverlay
              text={overlay.text}
              durationFrames={overlay.durationFrames}
              emphasis={overlay.emphasis}
            />
          </Sequence>
        ))}
      </AbsoluteFill>
    );
  }

  // Animation/overlay scenes: stack segment clips, then overlay avatar PIP.
  let frameOffset = 0;
  return (
    <AbsoluteFill style={{ backgroundColor: "#000" }}>
      {scene.segments.map((seg, i) => {
        const from = frameOffset;
        frameOffset += seg.durationFrames;
        return (
          <Sequence
            key={`seg-${i}`}
            from={from}
            durationInFrames={seg.durationFrames}
            premountFor={10}
          >
            <SegmentSlot seg={seg} />
          </Sequence>
        );
      })}

      {scene.voiceFile && <Audio src={staticFile(scene.voiceFile)} />}

      {/* Avatar PIP — bottom right */}
      {scene.avatarFile && (
        <div
          style={{
            position: "absolute",
            bottom: 32,
            right: 32,
            width: width * 0.22,
            height: height * 0.22,
            borderRadius: 18,
            overflow: "hidden",
            boxShadow: "0 12px 40px rgba(0,0,0,0.7)",
            border: "2px solid rgba(255,255,255,0.18)",
          }}
        >
          <OffthreadVideo
            src={staticFile(scene.avatarFile)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
            muted
          />
        </div>
      )}

      {scene.textOverlays.map((overlay, i) => (
        <Sequence
          key={`text-${i}`}
          from={overlay.startFrame}
          durationInFrames={overlay.durationFrames}
          premountFor={5}
        >
          <TextOverlay
            text={overlay.text}
            durationFrames={overlay.durationFrames}
            emphasis={overlay.emphasis}
          />
        </Sequence>
      ))}
    </AbsoluteFill>
  );
};
