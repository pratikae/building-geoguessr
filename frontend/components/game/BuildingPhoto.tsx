"use client";

import { useState } from "react";
import Image from "next/image";
import { ZoomIn, ZoomOut, X } from "lucide-react";
import { Building } from "@/lib/types";

interface BuildingPhotoProps {
  building: Building;
  showHeatmap?: boolean;
}

export default function BuildingPhoto({ building, showHeatmap = false }: BuildingPhotoProps) {
  const [lightboxOpen, setLightboxOpen] = useState(false);

  return (
    <>
      {/* Main photo panel */}
      <div className="relative w-full h-full overflow-hidden rounded-lg panel group">
        {/* Scanline overlay */}
        <div className="scanlines absolute inset-0 z-10 pointer-events-none" />

        {/* Corner decorations */}
        <div
          className="absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 z-20 pointer-events-none"
          style={{ borderColor: "#00d4ff" }}
        />
        <div
          className="absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 z-20 pointer-events-none"
          style={{ borderColor: "#00d4ff" }}
        />
        <div
          className="absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 z-20 pointer-events-none"
          style={{ borderColor: "#00d4ff" }}
        />
        <div
          className="absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 z-20 pointer-events-none"
          style={{ borderColor: "#00d4ff" }}
        />

        {/* Grad-CAM heatmap overlay */}
        {showHeatmap && (
          <div className="heatmap-overlay absolute inset-0 z-10 pointer-events-none mix-blend-screen" />
        )}

        {/* Image */}
        <Image
          src={building.imageUrl}
          alt={building.name}
          fill
          className="object-cover"
          sizes="(max-width: 768px) 100vw, 50vw"
          priority
          unoptimized
        />

        {/* Zoom button */}
        <button
          onClick={() => setLightboxOpen(true)}
          className="absolute bottom-3 right-3 z-20 p-2 rounded-md opacity-0 group-hover:opacity-100 transition-opacity"
          style={{
            background: "rgba(13, 13, 24, 0.9)",
            border: "1px solid rgba(0, 212, 255, 0.3)",
          }}
          title="Zoom in"
        >
          <ZoomIn size={16} style={{ color: "#00d4ff" }} />
        </button>

        {/* Year badge */}
        <div
          className="absolute top-3 left-3 z-20 px-2 py-1 rounded text-xs font-mono"
          style={{
            background: "rgba(13, 13, 24, 0.85)",
            border: "1px solid rgba(0, 212, 255, 0.3)",
            color: "#00d4ff",
            fontFamily: "var(--font-geist-mono)",
          }}
        >
          BUILT {building.yearBuilt}
        </div>
      </div>

      {/* Lightbox */}
      {lightboxOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(5, 5, 9, 0.95)" }}
          onClick={() => setLightboxOpen(false)}
        >
          <div
            className="relative w-full max-w-5xl max-h-screen"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setLightboxOpen(false)}
              className="absolute -top-10 right-0 p-1 rounded-full z-10"
              style={{ color: "#64748b" }}
            >
              <X size={24} />
            </button>
            <div className="relative w-full aspect-video rounded-lg overflow-hidden">
              <Image
                src={building.imageUrl}
                alt={building.name}
                fill
                className="object-contain"
                unoptimized
              />
            </div>
            <p
              className="mt-3 text-center text-sm"
              style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
            >
              {building.name} — {building.country} — {building.yearBuilt}
            </p>
          </div>
        </div>
      )}
    </>
  );
}
