"use client";

import dynamic from "next/dynamic";
import { GlobePin, GlobeArc } from "@/lib/types";

const Globe3D = dynamic(() => import("./Globe3D"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="flex flex-col items-center gap-3">
        <div className="w-12 h-12 rounded-full border-2 border-cyan border-t-transparent animate-spin" />
        <span
          className="text-xs text-cyan-dim font-mono tracking-widest"
          style={{ fontFamily: "var(--font-geist-mono)" }}
        >
          INITIALIZING GLOBE...
        </span>
      </div>
    </div>
  ),
});

interface GlobeWrapperProps {
  pins?: GlobePin[];
  arcs?: GlobeArc[];
  customPin?: import("@/lib/types").Coordinates | null;
  onGlobeClick?: (lat: number, lng: number) => void;
  autoRotate?: boolean;
  interactive?: boolean;
  focusLat?: number;
  focusLng?: number;
  altitude?: number;
}

export default function GlobeWrapper(props: GlobeWrapperProps) {
  return <Globe3D {...props} />;
}
