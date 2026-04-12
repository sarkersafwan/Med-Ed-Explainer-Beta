import { Composition } from "remotion";
import { MedicalVideo } from "./MedicalVideo";
import type { VideoProps } from "./types";

// Default props — will be overridden by --props flag during render
const defaultProps: VideoProps = {
  scenes: [],
  audioFile: "",
  fps: 30,
  totalDurationFrames: 300,
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="MedicalVideo"
      component={MedicalVideo as unknown as React.ComponentType<Record<string, unknown>>}
      durationInFrames={defaultProps.totalDurationFrames}
      fps={defaultProps.fps}
      width={1920}
      height={1080}
      defaultProps={defaultProps as unknown as Record<string, unknown>}
      calculateMetadata={async ({ props }) => {
        const videoProps = props as unknown as VideoProps;
        return {
          durationInFrames: videoProps.totalDurationFrames,
          fps: videoProps.fps,
        };
      }}
    />
  );
};
