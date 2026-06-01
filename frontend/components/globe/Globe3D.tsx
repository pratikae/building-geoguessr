"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import Globe, { GlobeMethods } from "react-globe.gl";
import { GlobePin, GlobeArc, Coordinates } from "@/lib/types";

interface Globe3DProps {
  pins?: GlobePin[];
  arcs?: GlobeArc[];
  customPin?: Coordinates | null;  // renders the custom pin.png image
  onGlobeClick?: (lat: number, lng: number) => void;
  autoRotate?: boolean;
  interactive?: boolean;
  focusLat?: number;
  focusLng?: number;
  altitude?: number;
}

export default function Globe3D({
  pins = [],
  arcs = [],
  customPin = null,
  onGlobeClick,
  autoRotate = false,
  interactive = true,
  focusLat,
  focusLng,
  altitude = 2.5,
}: Globe3DProps) {
  const globeRef = useRef<GlobeMethods | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ width: 800, height: 600 });

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const update = () => {
      const { width, height } = container.getBoundingClientRect();
      if (width > 0 && height > 0) setDims({ width, height });
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const globe = globeRef.current;
    if (!globe || dims.width === 0) return;
    const controls = globe.controls();
    controls.autoRotate = autoRotate;
    controls.autoRotateSpeed = 0.5;
    controls.enableZoom = interactive;
    controls.enableRotate = interactive;
    controls.enablePan = false;
    if (interactive) {
      controls.minDistance = 185;
      controls.maxDistance = 580;
    }
  }, [autoRotate, interactive, dims]);

  useEffect(() => {
    const globe = globeRef.current;
    if (!globe || focusLat === undefined || focusLng === undefined) return;
    globe.pointOfView({ lat: focusLat, lng: focusLng, altitude }, 1200);
  }, [focusLat, focusLng, altitude]);

  const handleClick = useCallback(
    ({ lat, lng }: { lat: number; lng: number }) => {
      onGlobeClick?.(lat, lng);
    },
    [onGlobeClick]
  );

  // Build an HTMLElement for the custom pin image.
  // translateX(-50%) centres horizontally on the click point.
  // translateY(-100%) raises the element so its BOTTOM (needle tip) sits exactly
  // at the globe surface coordinate rather than the element's centre.
  // Note: pin.png needs a transparent background for cleanest look on the dark globe.
  // mix-blend-mode:multiply makes white areas disappear on dark backgrounds.
  const makePinEl = useCallback(() => {
    const wrap = document.createElement("div");
    wrap.style.width = "28px";
    wrap.style.transform = "translateX(-50%) translateY(-100%)";
    wrap.style.pointerEvents = "none";
    const img = document.createElement("img");
    img.src = "/pin.png";
    img.style.width = "100%";
    img.style.height = "auto";
    img.style.mixBlendMode = "multiply";
    img.style.filter = "drop-shadow(0 2px 6px rgba(0,0,180,0.6)) brightness(1.3)";
    wrap.appendChild(img);
    return wrap;
  }, []);

  const htmlPinData = customPin ? [customPin] : [];

  return (
    <div ref={containerRef} style={{ width: "100%", height: "100%", overflow: "hidden" }}>
      <Globe
        ref={globeRef}
        width={dims.width}
        height={dims.height}
        globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
        bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
        backgroundColor="rgba(0,0,0,0)"
        showAtmosphere
        atmosphereColor="#d4a017"
        atmosphereAltitude={0.18}
        // Coloured dots — used on results screen for user / model / actual
        pointsData={pins}
        pointLat={(d) => (d as GlobePin).lat}
        pointLng={(d) => (d as GlobePin).lng}
        pointColor={(d) => (d as GlobePin).color}
        pointAltitude={0.02}
        pointRadius={(d) => (d as GlobePin).size}
        pointLabel={(d) => (d as GlobePin).label}
        // Custom HTML pin image — used during active guessing
        htmlElementsData={htmlPinData}
        htmlElement={makePinEl}
        htmlLat={(d) => (d as Coordinates).lat}
        htmlLng={(d) => (d as Coordinates).lng}
        htmlAltitude={0.01}
        // Arcs
        arcsData={arcs}
        arcStartLat={(d) => (d as GlobeArc).startLat}
        arcStartLng={(d) => (d as GlobeArc).startLng}
        arcEndLat={(d) => (d as GlobeArc).endLat}
        arcEndLng={(d) => (d as GlobeArc).endLng}
        arcColor={(d: object) => (d as GlobeArc).color}
        arcAltitude={0.35}
        arcStroke={0.6}
        arcDashLength={0.4}
        arcDashGap={0.2}
        arcDashAnimateTime={2500}
        onGlobeClick={handleClick}
        enablePointerInteraction={interactive}
      />
    </div>
  );
}
