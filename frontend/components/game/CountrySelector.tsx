"use client";

import { useState, useRef, useEffect } from "react";
import { Search, ChevronDown, MapPin } from "lucide-react";
import { COUNTRIES } from "@/lib/countries";
import { Country } from "@/lib/types";

const MONO = "var(--font-geist-mono)";

interface CountrySelectorProps {
  value: string | null;
  onChange: (country: Country) => void;
}

export default function CountrySelector({ value, onChange }: CountrySelectorProps) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const selected = COUNTRIES.find((c) => c.code === value);

  const filtered = query.trim()
    ? COUNTRIES.filter(
        (c) =>
          c.name.toLowerCase().includes(query.toLowerCase()) ||
          c.continent.toLowerCase().includes(query.toLowerCase())
      ).slice(0, 20)
    : COUNTRIES;

  const grouped = query.trim()
    ? null
    : filtered.reduce<Record<string, Country[]>>((acc, c) => {
        if (!acc[c.continent]) acc[c.continent] = [];
        acc[c.continent].push(c);
        return acc;
      }, {});

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function handleSelect(country: Country) {
    onChange(country);
    setQuery("");
    setOpen(false);
  }

  return (
    <div ref={containerRef} className="relative w-full">
      {/* Trigger */}
      <button
        onClick={() => {
          setOpen(!open);
          if (!open) setTimeout(() => inputRef.current?.focus(), 50);
        }}
        className="w-full flex items-center gap-3 text-left transition-colors"
        style={{
          padding: "14px 16px",
          borderRadius: 12,
          background: "#13131f",
          border: `1px solid ${selected ? "rgba(0,212,255,0.4)" : "rgba(0,212,255,0.15)"}`,
          color: selected ? "#e2e8f0" : "#64748b",
        }}
      >
        <MapPin size={18} style={{ color: "#00d4ff", flexShrink: 0 }} />
        <span className="flex-1 truncate" style={{ fontFamily: MONO, fontSize: "1.05rem" }}>
          {selected ? selected.name : "Select a country"}
        </span>
        {selected && (
          <span style={{ color: "#64748b", fontSize: "0.88rem", fontFamily: MONO }}>
            {selected.continent}
          </span>
        )}
        <ChevronDown
          size={18}
          style={{
            color: "#64748b",
            transform: open ? "rotate(180deg)" : "rotate(0deg)",
            transition: "transform 0.2s",
            flexShrink: 0,
          }}
        />
      </button>

      {/* Dropdown */}
      {open && (
        <div
          className="absolute bottom-full mb-2 left-0 right-0 z-30 rounded-2xl overflow-hidden"
          style={{
            background: "#0d0d18",
            border: "1px solid rgba(0,212,255,0.2)",
            boxShadow: "0 -16px 48px rgba(0,0,0,0.8)",
            maxHeight: "340px",
          }}
        >
          {/* Search */}
          <div
            className="flex items-center gap-3 sticky top-0"
            style={{
              padding: "14px 16px",
              background: "#0d0d18",
              borderBottom: "1px solid rgba(0,212,255,0.1)",
            }}
          >
            <Search size={17} style={{ color: "#64748b", flexShrink: 0 }} />
            <input
              ref={inputRef}
              type="text"
              placeholder="Search countries..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="flex-1 bg-transparent outline-none"
              style={{
                color: "#e2e8f0",
                fontFamily: MONO,
                fontSize: "1rem",
              }}
            />
          </div>

          {/* List */}
          <div className="scroll-area overflow-y-auto" style={{ maxHeight: "280px" }}>
            {query.trim() ? (
              filtered.length > 0 ? (
                filtered.map((c) => (
                  <CountryOption key={c.code} country={c} selected={c.code === value} onSelect={handleSelect} />
                ))
              ) : (
                <div className="text-center py-6" style={{ color: "#64748b", fontFamily: MONO, fontSize: "0.95rem" }}>
                  No countries found
                </div>
              )
            ) : (
              grouped &&
              Object.entries(grouped)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([continent, countries]) => (
                  <div key={continent}>
                    <div
                      className="uppercase tracking-widest sticky top-0"
                      style={{
                        padding: "10px 16px",
                        color: "#00d4ff",
                        background: "#0d0d18",
                        borderBottom: "1px solid rgba(0,212,255,0.06)",
                        fontFamily: MONO,
                        fontSize: "0.78rem",
                        letterSpacing: "0.18em",
                      }}
                    >
                      {continent}
                    </div>
                    {countries.map((c) => (
                      <CountryOption key={c.code} country={c} selected={c.code === value} onSelect={handleSelect} />
                    ))}
                  </div>
                ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function CountryOption({ country, selected, onSelect }: { country: Country; selected: boolean; onSelect: (c: Country) => void }) {
  return (
    <button
      onClick={() => onSelect(country)}
      className="w-full text-left flex items-center gap-3 transition-colors"
      style={{
        padding: "13px 16px",
        background: selected ? "rgba(0,212,255,0.08)" : "transparent",
        color: selected ? "#00d4ff" : "#e2e8f0",
        fontFamily: MONO,
        fontSize: "1rem",
      }}
      onMouseEnter={(e) => { if (!selected) (e.currentTarget as HTMLElement).style.background = "rgba(0,212,255,0.04)"; }}
      onMouseLeave={(e) => { if (!selected) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
    >
      <span style={{ color: "#4a5568", width: 40, flexShrink: 0, fontSize: "0.85rem" }}>{country.code}</span>
      <span>{country.name}</span>
      {selected && <span className="ml-auto" style={{ color: "#00d4ff" }}>&#10003;</span>}
    </button>
  );
}
