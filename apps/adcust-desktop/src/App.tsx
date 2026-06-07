import { useEffect, useMemo, useState } from "react";
import { Navigate, Route, Routes, Link, useLocation } from "react-router-dom";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  Activity,
  Cpu,
  Maximize2,
  Minus,
  PlugZap,
  RefreshCw,
  Server,
  Settings,
  Terminal,
  X,
  Zap,
} from "lucide-react";

import ComputeRegistry from "./pages/ComputeRegistry";
import Inference from "./pages/Inference";
import Training from "./pages/Training";

type BackendState = {
  status: "unconfigured" | "checking" | "online" | "offline";
  latencyMs?: number;
  message: string;
  checkedUrl?: string;
};

const BACKEND_URL_KEY = "adcust.backendBaseUrl";
const LOCAL_BACKEND_CANDIDATES = ["http://127.0.0.1:8000", "http://localhost:8000"];

function normalizeBackendBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, "").replace(/\/api\/v1$/i, "");
}

export default function App() {
  const location = useLocation();
  const [backendBaseUrl, setBackendBaseUrl] = useState(() => normalizeBackendBaseUrl(localStorage.getItem(BACKEND_URL_KEY) || ""));
  const [draftBackendUrl, setDraftBackendUrl] = useState(backendBaseUrl);
  const [backendState, setBackendState] = useState<BackendState>({
    status: "checking",
    message: "Checking backend",
  });

  const normalizedBackendUrl = useMemo(() => normalizeBackendBaseUrl(backendBaseUrl), [backendBaseUrl]);

  const navItems = [
    { path: "/", icon: <Server size={16} />, label: "Compute", tag: "NODE" },
    { path: "/training", icon: <Cpu size={16} />, label: "Training", tag: "TUNE" },
    { path: "/inference", icon: <Zap size={16} />, label: "Inference", tag: "EXEC" },
  ];

  const probeBackend = async (baseUrl: string) => {
    const started = performance.now();
    let response = await fetch(`${baseUrl}/health`, { method: "GET", cache: "no-store" }).catch(() => null);
    if (!response || !response.ok) {
      response = await fetch(`${baseUrl}/`, { method: "GET", cache: "no-store" });
    }
    const latencyMs = Math.round(performance.now() - started);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const data = await response.json().catch(() => ({}));
    return { latencyMs, data };
  };

  const checkBackend = async () => {
    const candidates = normalizedBackendUrl
      ? [normalizedBackendUrl, ...LOCAL_BACKEND_CANDIDATES.filter((url) => url !== normalizedBackendUrl)]
      : LOCAL_BACKEND_CANDIDATES;

    setBackendState((prev) => ({ ...prev, status: "checking", message: "Checking backend" }));
    const errors: string[] = [];

    for (const candidate of candidates) {
      try {
        const { latencyMs, data } = await probeBackend(candidate);
        if (candidate !== normalizedBackendUrl) {
          setBackendBaseUrl(candidate);
          setDraftBackendUrl(candidate);
          localStorage.setItem(BACKEND_URL_KEY, candidate);
        }
        setBackendState({
          status: "online",
          latencyMs,
          checkedUrl: candidate,
          message: data.status ? `Backend ${data.status}` : "Backend online",
        });
        return;
      } catch (error: any) {
        errors.push(`${candidate}: ${error.message || "unreachable"}`);
      }
    }
    setBackendState({
      status: normalizedBackendUrl ? "offline" : "unconfigured",
      message: errors.join(" | ") || "Backend unreachable",
    });
  };

  const saveBackendUrl = () => {
    const nextUrl = normalizeBackendBaseUrl(draftBackendUrl);
    setBackendBaseUrl(nextUrl);
    if (nextUrl) {
      localStorage.setItem(BACKEND_URL_KEY, nextUrl);
    } else {
      localStorage.removeItem(BACKEND_URL_KEY);
    }
  };

  const windowAction = async (action: "minimize" | "maximize" | "close") => {
    const win = getCurrentWindow();
    if (action === "minimize") await win.minimize();
    if (action === "maximize") await win.toggleMaximize();
    if (action === "close") await win.close();
  };

  useEffect(() => {
    checkBackend();
    const timer = window.setInterval(checkBackend, 5000);
    return () => window.clearInterval(timer);
  }, [normalizedBackendUrl]);

  const backendReady = backendState.status === "online";

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background text-on-surface select-none font-sans">
      <header data-tauri-drag-region className="h-12 shrink-0 border-b border-white/10 bg-[#101012] flex items-center justify-between">
        <div data-tauri-drag-region className="flex h-full min-w-0 flex-1 items-center">
          <div data-tauri-drag-region className="w-64 h-full border-r border-white/10 px-4 flex items-center gap-3">
            <div className="w-7 h-7 border border-primary/50 bg-primary/10 flex items-center justify-center">
              <Terminal size={15} className="text-primary" />
            </div>
            <div className="leading-none">
              <div className="text-sm font-black tracking-widest uppercase">AdCust</div>
              <div className="text-[9px] text-slate-500 uppercase tracking-wider">Adapter Customizer</div>
            </div>
          </div>

          <nav className="hidden md:flex h-full">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`h-full px-4 border-r border-white/5 flex items-center gap-2 text-xs transition-colors ${
                  location.pathname === item.path
                    ? "bg-primary/10 text-primary"
                    : "text-slate-400 hover:text-slate-100 hover:bg-white/5"
                }`}
              >
                {item.icon}
                <span className="font-bold">{item.label}</span>
                <span className="text-[9px] opacity-40">[{item.tag}]</span>
              </Link>
            ))}
          </nav>
        </div>

        <div className="h-full flex items-center border-l border-white/10">
          <div className="flex items-center gap-2 px-3 max-w-[520px]">
            <StatusDot status={backendState.status} />
            <span className="text-[10px] text-slate-300 font-mono uppercase truncate" title={`${backendState.message} ${backendState.checkedUrl || ""}`}>
              {backendState.message}
              {backendState.latencyMs !== undefined ? ` · ${backendState.latencyMs}ms` : ""}
              {backendState.checkedUrl ? ` · ${backendState.checkedUrl}` : ""}
            </span>
            <button title="Refresh backend status" onClick={checkBackend} className="icon-button">
              <RefreshCw size={13} />
            </button>
          </div>

          <button title="Minimize" onClick={() => windowAction("minimize")} className="window-button">
            <Minus size={14} />
          </button>
          <button title="Maximize" onClick={() => windowAction("maximize")} className="window-button">
            <Maximize2 size={13} />
          </button>
          <button title="Close" onClick={() => windowAction("close")} className="window-button window-button-danger">
            <X size={15} />
          </button>
        </div>
      </header>

      <div className="flex min-h-0 flex-1">
        <aside className="w-64 shrink-0 border-r border-white/10 bg-[#0d0d0f] flex flex-col">
          <section className="border-b border-white/10 p-4">
            <div className="flex items-center gap-2 text-xs font-black uppercase text-slate-300">
              <PlugZap size={15} className="text-primary" />
              Backend Endpoint
            </div>
            <div className="mt-3 flex gap-2">
              <input
                value={draftBackendUrl}
                onChange={(event) => setDraftBackendUrl(event.target.value)}
                onBlur={saveBackendUrl}
                onKeyDown={(event) => {
                  if (event.key === "Enter") saveBackendUrl();
                }}
                placeholder="Backend base URL"
                className="min-w-0 flex-1 bg-black/50 border border-white/20 px-2 h-8 text-[11px] font-mono outline-none focus:border-primary"
              />
              <button title="Save backend URL" onClick={saveBackendUrl} className="icon-button">
                <Settings size={13} className="mx-auto" />
              </button>
              <button title="Auto-detect local backend" onClick={checkBackend} className="icon-button">
                <RefreshCw size={13} className="mx-auto" />
              </button>
            </div>
            <div className="mt-2 text-[10px] font-mono text-slate-500 break-all">
              {backendState.checkedUrl || normalizedBackendUrl || "Auto-detects local backend on port 8000"}
            </div>
          </section>

          <nav className="flex-1 py-3 font-mono">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`group flex items-center px-4 py-2.5 text-[13px] transition-all border-l-2 ${
                  location.pathname === item.path
                    ? "border-primary bg-primary/10 text-primary"
                    : "border-transparent text-slate-400 hover:bg-white/5 hover:text-slate-100"
                }`}
              >
                <span className="mr-3">{item.icon}</span>
                <span className="flex-1 font-medium">{item.label}</span>
                <span className="text-[9px] opacity-35 group-hover:opacity-80">[{item.tag}]</span>
              </Link>
            ))}
          </nav>

          <section className="border-t border-white/10 p-4">
            <div className="text-[10px] text-slate-500 uppercase font-bold mb-3">Runtime Awareness</div>
            <div className="space-y-2 text-[10px] font-mono text-slate-400">
              <FeedLine ok={backendReady} text={backendReady ? "BACKEND_CONNECTED" : "BACKEND_NOT_READY"} />
              <FeedLine ok={Boolean(normalizedBackendUrl)} text={normalizedBackendUrl ? "API_URL_CONFIGURED" : "API_URL_REQUIRED"} />
            </div>
          </section>
        </aside>

        <main className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-surface">
          <div className="pointer-events-none absolute inset-0 pixel-grid opacity-15" />
          <div className="relative z-10 min-h-0 flex-1 overflow-y-auto custom-scrollbar p-6">
            <Routes>
              <Route path="/" element={<ComputeRegistry apiBase={normalizedBackendUrl} backendReady={backendReady} />} />
              <Route path="/training" element={<Training apiBase={normalizedBackendUrl} backendReady={backendReady} />} />
              <Route path="/inference" element={<Inference apiBase={normalizedBackendUrl} backendReady={backendReady} />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </div>
        </main>
      </div>

      <footer className="h-8 shrink-0 border-t border-white/10 bg-[#151518] flex items-center justify-between px-3">
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-500 uppercase">
          <Activity size={12} />
          <span>Desktop shell: Tauri</span>
        </div>
        <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 uppercase">
          <span>Backend: {backendState.status}</span>
          <span>Route: {location.pathname}</span>
        </div>
      </footer>
    </div>
  );
}

function StatusDot({ status }: { status: BackendState["status"] }) {
  const className =
    status === "online"
      ? "bg-secondary active-glow"
      : status === "checking"
        ? "bg-warning animate-pulse"
        : "bg-error";
  return <span className={`w-2 h-2 ${className}`} />;
}

function FeedLine({ ok, text }: { ok: boolean; text: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`w-1.5 h-1.5 ${ok ? "bg-secondary" : "bg-error"}`} />
      <span>{text}</span>
    </div>
  );
}
