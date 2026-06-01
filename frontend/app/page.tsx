"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { X, MapPin, Cpu, Trophy } from "lucide-react";

const MONO = "var(--font-geist-mono)";
const GOLD = "#d4a017";

const ANIM: Record<string, string> = {
  left:   "scrapFromLeft",
  right:  "scrapFromRight",
  top:    "scrapFromTop",
  bottom: "scrapFromBottom",
};

const SCRAP = [
  // FLOATING — sized to match the centre's scale
  { src: "/buildings6.png", w: 660, h: 435,
    css: { left: "1%", top: "30%", width: "36%" },
    rotate: -3.2, from: "left", delay: 0.0 },
  { src: "/buildings7.png", w: 900, h: 280,
    css: { right: "0%", top: "8%", width: "50%" },
    rotate: 2.0, from: "right", delay: 0.1 },
  // BOTTOM ROW — right to the bottom edge, black cut lines hidden
  { src: "/buildings2.png", w: 580, h: 720,
    css: { left: "-2%", bottom: "-12%", width: "26%" },
    rotate: 3.5, from: "bottom", delay: 0.2 },
  // Fill left gap: dome building tucked in
  { src: "/buildings4.png", w: 560, h: 460,
    css: { left: "8%", bottom: "-10%", width: "24%" },
    rotate: -2.2, from: "bottom", delay: 0.28 },
  // Fill further left gap: narrow tall portrait
  { src: "/buildings3.png", w: 580, h: 720,
    css: { left: "20%", bottom: "-18%", width: "20%" },
    rotate: 1.8, from: "bottom", delay: 0.35 },
  { src: "/buildings1.png", w: 900, h: 280,
    css: { left: "30%", bottom: "-3%", width: "46%" },
    rotate: -1.0, from: "bottom", delay: 0.4 },
  { src: "/buildings4.png", w: 560, h: 460,
    css: { right: "18%", bottom: "-14%", width: "25%" },
    rotate: 2.8, from: "bottom", delay: 0.5 },
  { src: "/buildings3.png", w: 580, h: 720,
    css: { right: "-3%", bottom: "-12%", width: "27%" },
    rotate: -4.0, from: "bottom", delay: 0.75 },
  { src: "/buildings5.png", w: 550, h: 850,
    css: { right: "3%", bottom: "-28%", width: "25%" },
    rotate: -3.2, from: "bottom", delay: 0.6 },
  { src: "/buildings8.png", w: 840, h: 320,
    css: { right: "-1%", bottom: "-4%", width: "40%" },
    rotate: 1.8, from: "bottom", delay: 0.65 },
];

const ALL_LANDED_MS = (0.75 + 0.92) * 1000 + 450;

function StartModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: "rgba(0,0,0,0.88)", backdropFilter: "blur(12px)" }}>
      <div className="relative rounded-3xl overflow-hidden fade-in-up"
        style={{ width: 520, background: "#0e0c09", border: "1px solid rgba(212,160,23,0.35)", boxShadow: "0 0 100px rgba(212,160,23,0.1)" }}>
        <div style={{ height: 2, background: "linear-gradient(90deg,transparent,#d4a017,transparent)" }} />
        <button onClick={onClose} className="absolute top-5 right-5"
          style={{ color: "#7a7060", background: "none", border: "none", cursor: "pointer" }}>
          <X size={22} />
        </button>
        <div className="flex flex-col items-center text-center gap-6" style={{ padding: "2.2rem 2.6rem 2.6rem" }}>
          <Image src="/bob_logo.png" alt="Bob" width={80} height={80} unoptimized style={{ borderRadius: 14, objectFit: "cover" }} />
          <div>
            <div style={{ color: GOLD, fontFamily: MONO, fontSize: "0.82rem", letterSpacing: "0.22em", marginBottom: 12 }}>YOUR OPPONENT</div>
            <h2 className="font-bold" style={{ fontSize: "2rem", color: "#f0ebe0", lineHeight: 1.12 }}>
              Ready to challenge<br />Bob the Builder?
            </h2>
          </div>
          <p style={{ color: "#7a7060", fontSize: "1rem", lineHeight: 1.75, maxWidth: 360 }}>
            Bob is an AI trained on 44,000 historic buildings. He will guess where every modern building is from. So will you.
          </p>
          <div className="w-full flex flex-col gap-2.5">
            {[
              { icon: <MapPin size={16} />, text: "5 rounds, one mystery building per round" },
              { icon: <Trophy size={16} />, text: "Drop a pin on the globe to place your guess" },
              { icon: <Cpu   size={16} />, text: "Bob reveals his analysis and guess after yours" },
            ].map(({ icon, text }, i) => (
              <div key={i} className="flex items-center gap-3 rounded-xl text-left"
                style={{ background: "#1a1a12", border: "1px solid rgba(212,160,23,0.1)", padding: "13px 16px" }}>
                <span style={{ color: GOLD, flexShrink: 0 }}>{icon}</span>
                <span style={{ color: "#a09282", fontFamily: MONO, fontSize: "0.9rem" }}>{text}</span>
              </div>
            ))}
          </div>
          <Link href="/game" className="w-full flex items-center justify-center rounded-2xl font-bold transition-all"
            style={{ padding: "18px", fontSize: "1.1rem", background: "linear-gradient(135deg,rgba(212,160,23,0.22),rgba(139,92,246,0.14))", border: "1px solid rgba(212,160,23,0.6)", color: GOLD, fontFamily: MONO, letterSpacing: "0.07em", boxShadow: "0 0 40px rgba(212,160,23,0.12)" }}
            onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 56px rgba(212,160,23,0.24)"; }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 40px rgba(212,160,23,0.12)"; }}>
            Let&apos;s go
          </Link>
        </div>
      </div>
    </div>
  );
}

function FlipCard({ onChallenge }: { onChallenge: () => void }) {
  const [flipped, setFlipped] = useState(false);

  const faceBase: React.CSSProperties = {
    position: "absolute",
    inset: 0,
    borderRadius: 28,
    backfaceVisibility: "hidden",
    WebkitBackfaceVisibility: "hidden" as React.CSSProperties["WebkitBackfaceVisibility"],
    background: "#111109",
    border: "1px solid rgba(212,160,23,0.38)",
    boxShadow: "0 0 80px rgba(0,0,0,0.85), 0 0 0 1px rgba(212,160,23,0.06)",
    overflow: "hidden",
    display: "flex",
    flexDirection: "column",
  };

  const goldBar = (
    <div style={{ height: 3, flexShrink: 0, background: "linear-gradient(90deg,transparent,#d4a017,transparent)" }} />
  );

  return (
    <div style={{ perspective: 1200, width: "clamp(500px, 36vw, 600px)", height: "clamp(640px, 90vh, 860px)" }}>
      <div style={{
        position: "relative", width: "100%", height: "100%",
        transformStyle: "preserve-3d",
        transition: "transform 0.75s cubic-bezier(0.4,0,0.2,1)",
        transform: flipped ? "rotateY(180deg)" : "rotateY(0deg)",
      }}>

        {/* FRONT */}
        <div style={{ ...faceBase, cursor: "pointer" }}
          onClick={() => setFlipped(true)} role="button" tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && setFlipped(true)}>
          {goldBar}
          <div style={{ padding: "1.4rem 1.8rem 1.6rem", display: "flex", flexDirection: "column", alignItems: "center", flex: 1, gap: "0.8rem", minHeight: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, alignSelf: "flex-start", flexShrink: 0 }}>
              <Image src="/bob_logo.png" alt="Beat Bob" width={40} height={40} unoptimized style={{ borderRadius: 10, objectFit: "cover" }} />
              <span style={{ color: "#f0ebe0", fontFamily: MONO, fontSize: "1.05rem", fontWeight: 700, letterSpacing: "0.08em" }}>Beat Bob</span>
            </div>
            <div style={{ flex: 1, position: "relative", width: "100%", display: "flex", alignItems: "center", justifyContent: "center", minHeight: 0 }}>
              <div style={{ position: "absolute", inset: 0, background: "radial-gradient(circle at 50% 60%, rgba(212,160,23,0.1) 0%, transparent 65%)" }} />
              <Image src="/bob_person.png" alt="Bob the Builder" width={300} height={415} unoptimized
                className="float"
                style={{ position: "relative", zIndex: 1, maxHeight: "100%", width: "auto", filter: "drop-shadow(0 16px 52px rgba(0,0,0,0.95))" }}
              />
            </div>
            <div style={{ flexShrink: 0, textAlign: "center" }}>
              <div style={{ color: GOLD, fontFamily: MONO, fontSize: "0.9rem", letterSpacing: "0.15em" }}>TAP TO MEET BOB</div>
              <div style={{ color: "#3a3628", fontFamily: MONO, fontSize: "0.76rem", marginTop: 4 }}>your AI opponent</div>
            </div>
          </div>
        </div>

        {/* BACK — click anywhere flips back to front */}
        <div style={{ ...faceBase, transform: "rotateY(180deg)", cursor: "pointer" }}
          onClick={() => setFlipped(false)}>
          {goldBar}
          <div style={{ padding: "1.8rem 2.2rem 1.8rem", display: "flex", flexDirection: "column", flex: 1, gap: "1rem", minHeight: 0, overflow: "hidden" }}>
            <div style={{ color: "rgba(212,160,23,0.55)", fontFamily: MONO, fontSize: "0.8rem", letterSpacing: "0.22em", flexShrink: 0 }}>
              THE QUESTION
            </div>
            <h2 className="font-bold" style={{ fontSize: "1.9rem", color: "#f0ebe0", lineHeight: 1.1, flexShrink: 0 }}>
              Every building used to tell you{" "}
              <span className="gradient-text">where it was from.</span>
            </h2>
            <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "0.9rem", overflow: "hidden" }}>
              <p style={{ color: "#7a7060", fontSize: "1.2rem", lineHeight: 1.78 }}>
                Regional craft and centuries of tradition were baked into every arch and gable.
                Then the world started sharing blueprints.
              </p>
              <p style={{ color: "#7a7060", fontSize: "1.2rem", lineHeight: 1.78 }}>
                <strong style={{ color: "#a09282" }}>Bob the Builder</strong> is our AI trained on
                44,000 historic buildings from 150+ countries. Can you beat him?
              </p>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10, flexShrink: 0 }}>
              <button
                onClick={(e) => { e.stopPropagation(); onChallenge(); }}
                style={{ width: "100%", padding: "18px", fontSize: "1.1rem", background: "linear-gradient(135deg,rgba(212,160,23,0.2),rgba(139,92,246,0.12))", border: "1px solid rgba(212,160,23,0.55)", color: GOLD, fontFamily: MONO, letterSpacing: "0.06em", boxShadow: "0 0 28px rgba(212,160,23,0.12)", cursor: "pointer", borderRadius: 16, fontWeight: 700 }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 44px rgba(212,160,23,0.22)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.boxShadow = "0 0 28px rgba(212,160,23,0.12)"; }}>
                Challenge Bob
              </button>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}

export default function Home() {
  const [ready, setReady] = useState(false);
  const [modal, setModal] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setReady(true), ALL_LANDED_MS);
    return () => clearTimeout(t);
  }, []);

  return (
    <main className="h-full w-full relative overflow-hidden" style={{ background: "#080806" }}>

      {SCRAP.map((p, i) => (
        <div key={i} className="absolute"
          style={{ ...p.css, transform: `rotate(${p.rotate}deg)`, zIndex: 10 } as React.CSSProperties}>
          <div style={{
            opacity: 0,
            animation: `${ANIM[p.from]} 0.85s cubic-bezier(0.22,1,0.36,1) forwards`,
            animationDelay: `${p.delay}s`,
            mixBlendMode: "lighten",
          } as React.CSSProperties}>
            <Image src={p.src} alt="" width={p.w} height={p.h}
              style={{ width: "100%", height: "auto" }} unoptimized priority={i < 3} />
          </div>
        </div>
      ))}

      <div className="absolute inset-0 flex items-center justify-center"
        style={{
          zIndex: 20,
          opacity: ready ? 1 : 0,
          transform: ready ? "translateY(0)" : "translateY(20px)",
          transition: "opacity 0.9s ease, transform 0.9s ease",
          pointerEvents: ready ? "auto" : "none",
        }}>
        <FlipCard onChallenge={() => setModal(true)} />
      </div>

      {modal && <StartModal onClose={() => setModal(false)} />}
    </main>
  );
}
