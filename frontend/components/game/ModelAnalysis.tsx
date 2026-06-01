"use client";

import { useEffect, useState } from "react";
import Image from "next/image";
import { Building, ModelPrediction, DetectedFeature } from "@/lib/types";

interface ModelAnalysisProps {
  building: Building;
  prediction: ModelPrediction;
  onComplete: () => void;
}

export default function ModelAnalysis({ building, prediction, onComplete }: ModelAnalysisProps) {
  const [phase, setPhase] = useState<"scanning" | "features" | "inference" | "done">("scanning");
  const [visibleFeatures, setVisibleFeatures] = useState<DetectedFeature[]>([]);
  const [continentBar, setContinentBar] = useState(0);
  const [countryBar, setCountryBar] = useState(0);
  const [scanPct, setScanPct] = useState(0);

  useEffect(() => {
    // Scanning phase with animated progress
    const scanInterval = setInterval(() => {
      setScanPct((p) => {
        if (p >= 100) {
          clearInterval(scanInterval);
          return 100;
        }
        return p + 2;
      });
    }, 30);

    const t1 = setTimeout(() => setPhase("features"), 1600);

    // Reveal features one by one
    prediction.features.forEach((feat, i) => {
      setTimeout(() => {
        setVisibleFeatures((prev) => [...prev, feat]);
      }, feat.delayMs);
    });

    const t2 = setTimeout(() => {
      setPhase("inference");
      // Animate confidence bars
      const barTimer = setInterval(() => {
        setContinentBar((p) => Math.min(p + 2, Math.round(prediction.confidence.continent * 100)));
        setCountryBar((p) => Math.min(p + 2, Math.round(prediction.confidence.country * 100)));
      }, 20);
      setTimeout(() => clearInterval(barTimer), 1500);
    }, prediction.features[prediction.features.length - 1]?.delayMs ?? 3000 + 800);

    const t3 = setTimeout(() => {
      setPhase("done");
      onComplete();
    }, (prediction.features[prediction.features.length - 1]?.delayMs ?? 3000) + 2400);

    return () => {
      clearInterval(scanInterval);
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
    };
  }, [prediction, onComplete]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: "rgba(5, 5, 9, 0.96)" }}
    >
      <div
        className="w-full max-w-5xl grid grid-cols-2 gap-4 rounded-xl overflow-hidden"
        style={{ border: "1px solid rgba(0, 212, 255, 0.2)", background: "#0d0d18" }}
      >
        {/* Left: building photo with scan effect */}
        <div className="relative aspect-square overflow-hidden">
          {/* Static background image */}
          <Image
            src={building.imageUrl}
            alt={building.name}
            fill
            className="object-cover"
            unoptimized
          />

          {/* Scanlines */}
          <div className="scanlines absolute inset-0 z-10 pointer-events-none" />

          {/* Grad-CAM heatmap (animates in) */}
          {phase !== "scanning" && (
            <div
              className="heatmap-overlay absolute inset-0 z-10 pointer-events-none mix-blend-screen"
              style={{
                background: `
                  radial-gradient(ellipse at 45% 38%, rgba(255,30,30,0.75) 0%, rgba(255,140,0,0.5) 18%, rgba(255,255,0,0.3) 36%, rgba(0,200,80,0.12) 55%, transparent 75%)
                `,
              }}
            />
          )}

          {/* Horizontal scan line */}
          {phase === "scanning" && (
            <div
              className="absolute left-0 right-0 h-0.5 z-20 pointer-events-none"
              style={{
                top: `${scanPct}%`,
                background: "linear-gradient(90deg, transparent, #00d4ff, transparent)",
                boxShadow: "0 0 12px #00d4ff",
                transition: "top 0.03s linear",
              }}
            />
          )}

          {/* Status label */}
          <div
            className="absolute bottom-0 left-0 right-0 z-20 px-4 py-3"
            style={{ background: "rgba(5, 5, 9, 0.8)" }}
          >
            <div
              className="text-xs font-mono tracking-widest"
              style={{ color: "#00d4ff", fontFamily: "var(--font-geist-mono)" }}
            >
              {phase === "scanning" && `Bob is scanning the image... ${scanPct}%`}
              {phase === "features" && "Bob is reading the architecture..."}
              {phase === "inference" && "Bob is making his guess..."}
              {phase === "done" && "Bob's done!"}
            </div>
          </div>
        </div>

        {/* Right: analysis readout */}
        <div className="flex flex-col p-5 gap-4">
          {/* Header */}
          <div>
            <div
              style={{
                color: "#64748b",
                fontFamily: "var(--font-geist-mono)",
                fontSize: "0.88rem",
                letterSpacing: "0.15em",
                marginBottom: 4,
              }}
            >
              BOB THE BUILDER IS THINKING
            </div>
            <div className="font-bold" style={{ color: "#e2e8f0", fontSize: "1.3rem" }}>
              What does Bob notice?
            </div>
          </div>

          {/* Feature detections */}
          <div className="flex-1">
            <div
              style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem", letterSpacing: "0.15em", marginBottom: 8 }}
            >
              CLUES BOB SPOTTED:
            </div>
            <div className="space-y-2">
              {prediction.features.map((feat) => {
                const visible = visibleFeatures.includes(feat);
                return (
                  <div
                    key={feat.label}
                    className="transition-all duration-500"
                    style={{ opacity: visible ? 1 : 0, transform: visible ? "translateX(0)" : "translateX(-8px)" }}
                  >
                    <div className="flex justify-between items-center mb-0.5">
                      <span
                        className="text-sm"
                        style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}
                      >
                        {feat.label}
                      </span>
                      <span
                        className="text-xs font-mono"
                        style={{ color: "#00d4ff", fontFamily: "var(--font-geist-mono)" }}
                      >
                        {Math.round(feat.confidence * 100)}%
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div
                        className="flex-1 h-1 rounded-full overflow-hidden"
                        style={{ background: "rgba(0, 212, 255, 0.1)" }}
                      >
                        {visible && (
                          <div
                            className="h-full rounded-full transition-all duration-1000"
                            style={{
                              width: `${Math.round(feat.confidence * 100)}%`,
                              background: "linear-gradient(90deg, #7c3aed, #00d4ff)",
                            }}
                          />
                        )}
                      </div>
                      <span
                        className="text-xs truncate max-w-[120px]"
                        style={{ color: "#e2e8f0", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem" }}
                      >
                        {feat.value}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Inference section */}
          {(phase === "inference" || phase === "done") && (
            <div
              className="rounded-lg p-4 space-y-3"
              style={{ background: "#13131f", border: "1px solid rgba(0, 212, 255, 0.15)" }}
            >
              <div
                style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)", fontSize: "0.88rem", letterSpacing: "0.15em", marginBottom: 8 }}
              >
                BOB&apos;S CONCLUSION:
              </div>

              {/* Continent */}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm" style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}>
                    CONTINENT
                  </span>
                  <span className="text-xs font-mono" style={{ color: "#00d4ff", fontFamily: "var(--font-geist-mono)" }}>
                    {continentBar}%
                  </span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: "rgba(0, 212, 255, 0.1)" }}>
                  <div
                    className="h-full rounded-full transition-all duration-75"
                    style={{
                      width: `${continentBar}%`,
                      background: "linear-gradient(90deg, #7c3aed, #00d4ff)",
                      boxShadow: "0 0 8px rgba(0, 212, 255, 0.4)",
                    }}
                  />
                </div>
                <div className="mt-1 text-xs" style={{ color: "#e2e8f0", fontFamily: "var(--font-geist-mono)" }}>
                  â†’ {prediction.continent}
                </div>
              </div>

              {/* Country */}
              <div>
                <div className="flex justify-between mb-1">
                  <span className="text-sm" style={{ color: "#64748b", fontFamily: "var(--font-geist-mono)" }}>
                    COUNTRY
                  </span>
                  <span className="text-xs font-mono" style={{ color: "#00d4ff", fontFamily: "var(--font-geist-mono)" }}>
                    {countryBar}%
                  </span>
                </div>
                <div className="h-2 rounded-full overflow-hidden" style={{ background: "rgba(0, 212, 255, 0.1)" }}>
                  <div
                    className="h-full rounded-full transition-all duration-75"
                    style={{
                      width: `${countryBar}%`,
                      background: "linear-gradient(90deg, #10b981, #00d4ff)",
                      boxShadow: "0 0 8px rgba(16, 185, 129, 0.4)",
                    }}
                  />
                </div>
                <div className="mt-1 text-xs" style={{ color: "#e2e8f0", fontFamily: "var(--font-geist-mono)" }}>
                  â†’ {prediction.country}
                </div>
              </div>
            </div>
          )}

          {/* Pulsing status indicator */}
          <div className="flex items-center gap-2.5">
            <div
              className="w-2.5 h-2.5 rounded-full pulse-dot"
              style={{ background: phase === "done" ? "#10b981" : "#00d4ff" }}
            />
            <span
              style={{
                color: "#64748b",
                fontFamily: "var(--font-geist-mono)",
                fontSize: "0.93rem",
                letterSpacing: "0.15em",
              }}
            >
              {phase === "done" ? "BOB'S PREDICTION READY" : "BOB IS THINKING..."}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

