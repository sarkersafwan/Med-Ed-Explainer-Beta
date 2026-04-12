import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import type { OverlayEmphasis } from "./types";

interface Props {
  text: string;
  durationFrames: number;
  emphasis?: OverlayEmphasis;
}

/**
 * Split a clinical line on its first arrow / equals so cause and effect
 * land on different lines and the relationship reads instantly.
 */
function splitClinical(text: string): { lhs: string; sep: string; rhs: string } | null {
  const m = text.match(/^(.+?)\s*(→|->|=>|=|:)\s*(.+)$/);
  if (!m) return null;
  return { lhs: m[1].trim(), sep: m[2] === "->" ? "→" : m[2], rhs: m[3].trim() };
}

/**
 * Detect "key words" worth highlighting in yellow — clinical terms,
 * numbers with units, and ALL-CAPS tokens.
 */
const KEY_WORD_PATTERN = /(\b[A-Z]{2,}\b|\b\d+(?:\.\d+)?\s?(?:mm|cm|mg|kg|ml|mmol|mEq|mmHg|%|bpm)?\b|\bstops?\b|\bblocks?\b|\binhibits?\b|\bcauses?\b|\bprevents?\b|\bhypoxia\b|\bacidosis\b|\bischemia\b|\bnecrosis\b|\binfarct\w*\b|\bmechanism\b|\bpathway\b)/i;

const SOCIAL_FONT =
  "'Inter', 'SF Pro Display', 'Helvetica Neue', Helvetica, Arial, sans-serif";

// Heavy outline + drop shadow — this is what actually makes text legible
// over any background without needing a card. WebkitTextStroke handles the
// outline, filter: drop-shadow handles the lift.
const SOCIAL_STROKE_STYLE: React.CSSProperties = {
  WebkitTextStroke: "4px #000",
  textShadow:
    "0 4px 16px rgba(0,0,0,0.75), 0 2px 4px rgba(0,0,0,0.9), 0 0 2px #000",
  paintOrder: "stroke fill",
};

/** Render one word with optional yellow highlight if it matches a key term. */
const SocialWord: React.FC<{
  word: string;
  delayFrames: number;
  fontSize: number;
  isHighlight: boolean;
  forceColor?: string;
}> = ({ word, delayFrames, fontSize, isHighlight, forceColor }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const localFrame = frame - delayFrames;

  // Spring pop-in per word
  const enter = spring({
    frame: Math.max(0, localFrame),
    fps,
    config: { damping: 11, stiffness: 220 },
    durationInFrames: 12,
  });
  const opacity = interpolate(enter, [0, 1], [0, 1]);
  const scale = interpolate(enter, [0, 1], [0.4, 1]);
  const ty = interpolate(enter, [0, 1], [18, 0]);

  return (
    <span
      style={{
        display: "inline-block",
        fontSize,
        fontWeight: 900,
        fontFamily: SOCIAL_FONT,
        letterSpacing: -0.5,
        lineHeight: 1.05,
        color: forceColor ?? (isHighlight ? "#FFD84A" : "#FFFFFF"),
        ...SOCIAL_STROKE_STYLE,
        opacity,
        transform: `translateY(${ty}px) scale(${scale})`,
        marginRight: "0.28em",
        marginBottom: "0.05em",
      }}
    >
      {word}
    </span>
  );
};

const SocialLine: React.FC<{
  text: string;
  startFrame: number;
  fontSize: number;
  color?: string;
}> = ({ text, startFrame, fontSize, color }) => {
  const words = text.split(/\s+/).filter(Boolean);
  // 2 frames stagger between words — fast enough to feel energetic, slow
  // enough that each word is visible as it lands.
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "center",
        alignItems: "baseline",
      }}
    >
      {words.map((w, i) => (
        <SocialWord
          key={`${w}-${i}`}
          word={w}
          delayFrames={startFrame + i * 2}
          fontSize={fontSize}
          isHighlight={color ? false : KEY_WORD_PATTERN.test(w)}
          forceColor={color}
        />
      ))}
    </div>
  );
};

export const TextOverlay: React.FC<Props> = ({
  text,
  durationFrames,
  emphasis = "label",
}) => {
  const frame = useCurrentFrame();
  const { fps, width } = useVideoConfig();
  const fadeOut = Math.round(0.35 * fps);

  // Whole-overlay fade out at the end (individual word entrances handle the
  // in-animation). No fade-in at the container level to keep word pops crisp.
  const opacity = interpolate(
    frame,
    [durationFrames - fadeOut, durationFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const isClinical = emphasis === "clinical";
  const split = isClinical ? splitClinical(text) : null;

  // Social-safe placement: clinical sits in the upper-middle (avoids the
  // TikTok/Reels caption rail at the bottom and the notch at the top);
  // label callouts sit slightly lower at ~35% from bottom.
  const containerStyle: React.CSSProperties = {
    position: "absolute",
    top: isClinical ? "22%" : undefined,
    bottom: isClinical ? undefined : "32%",
    left: "6%",
    right: "6%",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    opacity,
    pointerEvents: "none",
    textAlign: "center",
    maxWidth: width * 0.88,
    margin: "0 auto",
  };

  // Sizes tuned for 1080p social — legible on a phone at arm's length.
  const clinicalMain = 92;
  const clinicalSep = 72;
  const labelSize = 78;

  return (
    <div style={containerStyle}>
      {split ? (
        <>
          <SocialLine text={split.lhs} startFrame={0} fontSize={clinicalMain} />
          <div style={{ height: 6 }} />
          <span
            style={{
              fontSize: clinicalSep,
              fontFamily: SOCIAL_FONT,
              fontWeight: 900,
              color: "#7FB8FF",
              ...SOCIAL_STROKE_STYLE,
              lineHeight: 1,
              marginBlock: 6,
            }}
          >
            {split.sep}
          </span>
          <div style={{ height: 6 }} />
          <SocialLine
            text={split.rhs}
            startFrame={Math.round(split.lhs.split(/\s+/).length * 2 + 6)}
            fontSize={clinicalMain}
            color="#FFD84A"
          />
        </>
      ) : (
        <SocialLine text={text} startFrame={0} fontSize={labelSize} />
      )}
    </div>
  );
};
