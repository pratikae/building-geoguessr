"use client";

import { useState, useEffect, useRef } from "react";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight } from "lucide-react";

const MONO = "var(--font-geist-mono)";
// Bob the Builder's palette — overalls blue + plaid-shirt orange
const BOB_BLUE = "#4a9eff";
const BOB_ORANGE = "#ff8c42";

const BRIEFING_LINES = [
"Every building used to tell you where it was from.",
"Globalization changed that. Same materials, same styles, same look everywhere.",
"Bob is an AI trained on 44,000 historic buildings from 150+ countries.",
"He'll guess where a modern building stands. So will you.",
"Five rounds. One pin each. Beat Bob."
];

function TypedLines({ lines, onDone }: { lines: string[]; onDone: () => void }) {
  const [lineIdx, setLineIdx] = useState(0);
  const [charIdx, setCharIdx] = useState(0);
  const fired = useRef(false);

  useEffect(() => {
    if (lineIdx >= lines.length) {
      if (!fired.current) { fired.current = true; onDone(); }
      return;
    }
    const current = lines[lineIdx];
    if (charIdx < current.length) {
      const t = setTimeout(() => setCharIdx((c) => c + 1), 26);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => { setLineIdx((l) => l + 1); setCharIdx(0); }, 650);
    return () => clearTimeout(t);
  }, [lineIdx, charIdx, lines, onDone]);

  return (
    <div className="flex flex-col gap-2" style={{ fontFamily: MONO, fontSize: "clamp(0.92rem, 2.2vh, 1.15rem)", lineHeight: 1.55, color: BOB_ORANGE, textShadow: "0 0 10px rgba(255,140,66,0.4)", textAlign: "left" }}>
      {lines.slice(0, lineIdx).map((l, i) => (
        <p key={i} style={{ margin: 0 }}><span style={{ color: BOB_BLUE }}>&gt;</span> {l}</p>
      ))}
      {lineIdx < lines.length && (
        <p style={{ margin: 0 }}>
          <span style={{ color: BOB_BLUE }}>&gt;</span> {lines[lineIdx].slice(0, charIdx)}
          <span style={{ display: "inline-block", color: BOB_BLUE, animation: "blink 1s step-end infinite" }}>█</span>
        </p>
      )}
    </div>
  );
}

type Phase = "hero" | "prompt" | "briefing";

// How wide Bob should read relative to the video's own pixel grid — tuned once at a
// reference size, then carried by `coverScale` to every viewport size/aspect ratio.
const BOB_REFERENCE_PX = 850;

export default function Home() {
  const [phase, setPhase] = useState<Phase>("hero");
  const [briefingDone, setBriefingDone] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [coverScale, setCoverScale] = useState(1);

  const revealOverlay = (next: Phase) => phase === "hero" && setPhase(next);

  // Bob must mask the footage's watermark, which scales with the video's own
  // object-cover crop — not with the viewport directly. So we measure the actual
  // cover scale factor (max of width-fit / height-fit ratios, exactly like `cover`
  // computes it) and size Bob off that, keeping coverage correct at any window
  // size or aspect ratio, including minimized/odd-shaped windows.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const recalc = () => {
      if (!video.videoWidth || !video.videoHeight) return;
      const scale = Math.max(
        window.innerWidth / video.videoWidth,
        window.innerHeight / video.videoHeight
      );
      setCoverScale(scale);
    };

    video.addEventListener("loadedmetadata", recalc);
    window.addEventListener("resize", recalc);
    recalc();

    return () => {
      video.removeEventListener("loadedmetadata", recalc);
      window.removeEventListener("resize", recalc);
    };
  }, []);

  return (
    <main className="h-full w-full relative overflow-hidden" style={{ background: "#080806" }}>

      {/* Looping video background */}
      <video
        ref={videoRef}
        autoPlay
        loop
        muted
        playsInline
        className="absolute inset-0 w-full h-full object-cover"
        style={{ zIndex: 0 }}
      >
        <source src="/bob_better.mp4" type="video/mp4" />
      </video>

      {/* Bob bursting through — masks the footage's watermark + click target.
          Sized off the video's measured cover-scale so it always covers, at any window size. */}
      <div
        onClick={() => revealOverlay("prompt")}
        role="button"
        tabIndex={0}
        aria-label="Meet Bob the Builder"
        onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && phase === "hero") { e.preventDefault(); setPhase("prompt"); } }}
        className="absolute inset-0 flex items-center justify-center"
        style={{ zIndex: 2, cursor: phase === "hero" ? "pointer" : "default", outline: "none" }}
      >
        <Image
          src="/Bob_Out_Of_Globe.png"
          alt="Bob the Builder bursting through the globe"
          width={6000}
          height={3375}
          unoptimized
          priority
          className="float"
          style={{ width: `${Math.round(coverScale * BOB_REFERENCE_PX)}px`, height: "auto", filter: "drop-shadow(0 24px 70px rgba(0,0,0,0.65))" }}
        />
      </div>

      {/* Onboarding sequence — hard cut to black, everything renders like an old terminal/CRT briefing.
          Content is anchored to a fixed top-left point (not centered) so each typed line lands in
          its final spot and stays put — like writing on a page — instead of re-centering as it grows. */}
      {phase !== "hero" && (
        <div className="absolute inset-0" style={{ zIndex: 10, background: "#000" }}>
          <div className="scanlines absolute inset-0 pointer-events-none" />

          {phase === "prompt" && (
            <div className="absolute inset-0 flex items-center justify-center">
              <button
                onClick={() => setPhase("briefing")}
                className="transition-colors"
                style={{
                  padding: "16px 40px", fontSize: "1.05rem", background: "transparent",
                  border: `1px solid ${BOB_BLUE}80`, borderRadius: 4,
                  color: BOB_ORANGE, fontFamily: MONO, letterSpacing: "0.18em",
                  textShadow: "0 0 10px rgba(255,140,66,0.4)", boxShadow: `0 0 28px ${BOB_BLUE}30`,
                  cursor: "pointer",
                }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = `${BOB_BLUE}18`; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}
              >
                <span style={{ color: BOB_BLUE }}>&gt;</span> GET STARTED_
              </button>
            </div>
          )}

          {phase === "briefing" && (
            <>
              {/* Bob — vertically centered on the right edge, close beside the brief */}
              <div className="absolute" style={{ right: "8%", top: "50%", transform: "translateY(-50%)", zIndex: 1 }}>
                <Image
                  src="/bob_person.png"
                  alt="Bob the Builder"
                  width={1124}
                  height={1435}
                  unoptimized
                  priority
                  className="float"
                  style={{ height: "min(56vh, 520px)", width: "auto", filter: "drop-shadow(0 24px 70px rgba(0,0,0,0.7))" }}
                />
              </div>

              {/* Briefing copy — pulled in next to Bob and roughly vertically centered.
                  `top` is offset by a fixed pixel amount (not 50% + transform) so the
                  anchor point itself never moves as lines type in — only the content
                  below it grows, keeping already-typed lines planted in place. */}
              <div className="absolute" style={{ top: "calc(50% - 185px)", left: "12%", width: "min(580px, 40vw)", maxHeight: "78vh", zIndex: 2 }}>
                <div className="flex flex-col gap-4">
                  <div style={{ color: BOB_BLUE, opacity: 0.7, fontFamily: MONO, fontSize: "0.74rem", letterSpacing: "0.32em" }}>
                    — INCOMING TRANSMISSION —
                  </div>
                  <TypedLines lines={BRIEFING_LINES} onDone={() => setBriefingDone(true)} />
                  {briefingDone && (
                    <Link href="/game" className="self-start flex items-center gap-3 transition-colors"
                      style={{
                        padding: "12px 30px", fontSize: "0.9rem", background: "transparent",
                        border: `1px solid ${BOB_BLUE}80`, borderRadius: 4,
                        color: BOB_ORANGE, fontFamily: MONO, letterSpacing: "0.14em",
                        textShadow: "0 0 10px rgba(255,140,66,0.4)", boxShadow: `0 0 28px ${BOB_BLUE}30`,
                      }}
                      onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = `${BOB_BLUE}18`; }}
                      onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "transparent"; }}>
                      <span style={{ color: BOB_BLUE }}>&gt;</span> LET&apos;S GO <ArrowRight size={16} />
                    </Link>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </main>
  );
}
