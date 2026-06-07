import { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { open } from "@tauri-apps/plugin-dialog";
import { AnimatePresence, motion } from "framer-motion";
import { Activity, AlertCircle, ChevronRight, Columns, FolderOpen, RotateCcw, Send, Square, Trash2, Unplug, XCircle, XOctagon } from "lucide-react";

type PageProps = {
  apiBase: string;
  backendReady: boolean;
};

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatHistory = {
  alpha: ChatMessage[];
  beta: ChatMessage[];
};

const DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Use the conversation history when answering.";

type AdapterData = {
  name: string;
  path: string;
};

type EngineStatus = "off" | "booting" | "running" | "generating";
type AdapterMountStatus = "unmounted" | "mounted" | "failed";

export default function Inference({ apiBase, backendReady }: PageProps) {
  const inferenceApi = useMemo(() => (apiBase ? `${apiBase}/api/v1/inference` : ""), [apiBase]);
  const sessionIdRef = useRef(`inference-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("off");
  const [isDualMode, setIsDualMode] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [modelPath, setModelPath] = useState("");
  const [maxTokens, setMaxTokens] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatHistory>({ alpha: [], beta: [] });
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const explainAxiosError = (errorValue: any) => {
    return errorValue.response?.data?.message || errorValue.response?.data?.detail || errorValue.message;
  };

  const selectModelPath = async () => {
    const selected = await open({
      multiple: false,
      directory: true,
      title: "Select local HuggingFace model folder",
    });
    if (selected && typeof selected === "string") setModelPath(selected);
  };

  const controlEngine = async (action: "boot" | "shutdown") => {
    try {
      setError(null);
      if (!inferenceApi) throw new Error("Backend base URL is not configured");
      if (action === "boot") {
        if (!modelPath) throw new Error("Select a local base model folder first");
        setEngineStatus("booting");
        await axios.post(`${inferenceApi}/engine/boot`, null, { params: { model_path: modelPath } });
        setEngineStatus("running");
      } else {
        await axios.post(`${inferenceApi}/engine/stop-generation`, { slot_id: null }).catch(() => undefined);
        abortControllerRef.current?.abort();
        abortControllerRef.current = null;
        setEngineStatus("off");
        await axios.post(`${inferenceApi}/engine/shutdown`);
        setChatHistory({ alpha: [], beta: [] });
      }
    } catch (errorValue: any) {
      setError(explainAxiosError(errorValue));
      setEngineStatus("off");
    }
  };

  const handleSendMessage = async () => {
    if (!prompt.trim() || engineStatus !== "running") return;
    if (!maxTokens) return setError("Max tokens is required");
    const userQuery = prompt;
    const historySnapshot = chatHistory;
    setPrompt("");
    setEngineStatus("generating");
    setChatHistory((prev) => ({
      alpha: [...prev.alpha, { role: "user", content: userQuery }, { role: "assistant", content: "" }],
      beta: isDualMode ? [...prev.beta, { role: "user", content: userQuery }, { role: "assistant", content: "" }] : prev.beta,
    }));

    abortControllerRef.current = new AbortController();
    try {
      const response = await fetch(`${inferenceApi}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          prompt: userQuery,
          session_id: sessionIdRef.current,
          system_prompt: systemPrompt,
          targets: isDualMode
            ? [
                { id: "alpha", history: historySnapshot.alpha },
                { id: "beta", history: historySnapshot.beta },
              ]
            : [{ id: "alpha", history: historySnapshot.alpha }],
          strategy: "interleaved",
          max_tokens: Number(maxTokens),
        }),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      if (!response.body) throw new Error("Inference stream is unavailable");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let lineBuffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        lineBuffer += decoder.decode(value, { stream: true });
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() || "";
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;
          const data = JSON.parse(trimmedLine);
          if (data.error) {
            setError(data.error);
          }
          const targetKey = data.source === "beta" && isDualMode ? "beta" : "alpha";
          setChatHistory((prev) => {
            const updatedList = [...prev[targetKey]];
            const lastIndex = updatedList.length - 1;
            if (lastIndex >= 0) {
              updatedList[lastIndex] = {
                ...updatedList[lastIndex],
                content: data.error ? `Runtime error: ${data.error}` : updatedList[lastIndex].content + (data.token || ""),
              };
            }
            return { ...prev, [targetKey]: updatedList };
          });
        }
      }
    } catch (errorValue: any) {
      if (errorValue.name !== "AbortError") setError(`Inference error: ${errorValue.message}`);
    } finally {
      abortControllerRef.current = null;
      setEngineStatus((prev) => (prev === "generating" ? "running" : prev));
    }
  };

  const stopGeneration = async () => {
    if (inferenceApi) {
      await axios.post(`${inferenceApi}/engine/stop-generation`, { slot_id: null }).catch((errorValue) => {
        setError(explainAxiosError(errorValue));
      });
    }
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setEngineStatus("running");
  };

  const clearHistory = async () => {
    setChatHistory({ alpha: [], beta: [] });
    if (!inferenceApi) return;
    await axios.post(`${inferenceApi}/chat/history/clear`, { session_id: sessionIdRef.current }).catch((errorValue) => {
      setError(explainAxiosError(errorValue));
    });
  };

  const engineOnline = engineStatus === "running" || engineStatus === "generating";

  return (
    <div className="flex flex-col h-full bg-cyber-black relative text-sm">
      <header className="h-16 border-b border-white/10 flex items-center justify-between px-5 bg-surface-container-low z-10">
        <div className="flex gap-4 items-center">
          <Activity className={engineOnline ? "text-secondary" : "text-error"} size={17} />
          <div>
            <div className="font-bold">Local Inference Lab</div>
            <div className={`text-[11px] font-mono uppercase ${engineOnline ? "text-secondary" : "text-error"}`}>
              Base model: {engineOnline ? "running" : engineStatus === "booting" ? "booting" : "stopped"}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={selectModelPath} className="btn-tool max-w-[280px]">
            <FolderOpen size={13} />
            <span className="truncate">{modelPath || "Select local model"}</span>
          </button>
          <input value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} type="number" placeholder="Max tokens" className="w-28 h-9 bg-black/40 border border-white/20 px-2 text-[11px] outline-none focus:border-primary" />
          <div className="flex bg-black/40 p-1 border border-white/10">
            <button title="Single adapter stream" onClick={() => setIsDualMode(false)} className={`btn-tool h-8 min-h-8 px-3 ${!isDualMode ? "border-primary bg-primary/20 text-white" : ""}`}><Square size={11} /></button>
            <button title="Dual adapter stream" onClick={() => setIsDualMode(true)} className={`btn-tool h-8 min-h-8 px-3 ${isDualMode ? "border-primary bg-primary/20 text-white" : ""}`}><Columns size={11} /></button>
          </div>
          <button
            onClick={() => controlEngine(engineStatus === "off" ? "boot" : "shutdown")}
            disabled={!backendReady || engineStatus === "booting"}
            className={engineOnline ? "btn-danger" : "btn-primary"}
          >
            {engineStatus === "off" ? "BOOT_ENGINE" : engineStatus === "booting" ? "BOOTING" : "SHUTDOWN_ENGINE"}
          </button>
        </div>
      </header>

      <section className="border-b border-white/10 bg-black/30 px-5 py-3">
        <div className="flex items-start gap-3">
          <div className="min-w-32 pt-1">
            <div className="text-[10px] font-black uppercase text-white/50">System Prompt</div>
            <div className="text-[10px] text-white/35">Editable session rule</div>
          </div>
          <textarea
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            className="min-h-16 flex-1 resize-y bg-black/50 border border-white/20 px-3 py-2 text-xs leading-relaxed text-white outline-none focus:border-primary focus:bg-black"
            placeholder="Optional system prompt"
          />
          <div className="flex flex-col gap-2">
            <button type="button" className="btn-tool h-8 min-h-8" onClick={() => setSystemPrompt(DEFAULT_SYSTEM_PROMPT)}>
              <RotateCcw size={13} /> Default
            </button>
            <button type="button" className="btn-danger h-8 min-h-8" onClick={clearHistory}>
              <Trash2 size={13} /> Clear History
            </button>
          </div>
        </div>
      </section>

      <div className="flex-1 flex gap-px bg-cyber-green/5 overflow-hidden">
        <ChatPanel apiBase={inferenceApi} id="ALPHA" engineStatus={engineStatus} modelPath={modelPath} messages={chatHistory.alpha} />
        {isDualMode && <ChatPanel apiBase={inferenceApi} id="BETA" engineStatus={engineStatus} modelPath={modelPath} messages={chatHistory.beta} />}
      </div>

      <div className="p-5 bg-surface-container-low border-t border-white/10">
        <div className={`relative flex items-center border transition-all ${engineOnline ? "border-secondary/50 bg-black" : "border-error/40 bg-black/40"}`}>
          <div className={`pl-4 pr-2 ${engineOnline ? "text-secondary" : "text-error"}`}><ChevronRight size={18} /></div>
          <input
            disabled={engineStatus !== "running"}
            className="flex-1 bg-transparent border-none outline-none text-on-surface py-4 text-sm font-medium"
            placeholder={engineStatus === "running" ? "Type a prompt" : "Engine is offline"}
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => event.key === "Enter" && handleSendMessage()}
          />
          <button disabled={engineStatus !== "running" || !prompt.trim()} onClick={handleSendMessage} className="btn-tool h-14 min-h-14 px-4 border-y-0 border-r-0">
            <Send size={20} />
          </button>
          <button disabled={engineStatus !== "generating"} onClick={stopGeneration} className="btn-danger h-14 min-h-14 px-4 border-y-0 border-r-0" title="Stop current generation">
            <XOctagon size={20} />
          </button>
        </div>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="absolute bottom-32 left-6 right-6 z-50 p-4 bg-red-950 border-l-4 border-red-500 flex items-start gap-4 shadow-2xl">
            <AlertCircle className="text-red-500 shrink-0 mt-0.5" size={18} />
            <div className="flex-1 overflow-hidden">
              <h4 className="text-[10px] font-black text-red-500 uppercase mb-1">Runtime Error</h4>
              <p className="text-xs text-red-200 font-bold truncate">{error}</p>
            </div>
            <button onClick={() => setError(null)} className="text-red-500 hover:text-white"><XCircle size={18} /></button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ChatPanel({
  apiBase,
  id,
  engineStatus,
  modelPath,
  messages,
}: {
  apiBase: string;
  id: string;
  engineStatus: EngineStatus;
  modelPath: string;
  messages: ChatMessage[];
}) {
  const [adapterInfo, setAdapterInfo] = useState<AdapterData | null>(null);
  const [adapterError, setAdapterError] = useState<string | null>(null);
  const [mountStatus, setMountStatus] = useState<AdapterMountStatus>("unmounted");
  const [mountMessage, setMountMessage] = useState("No adapter mounted.");
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const wasRunningRef = useRef(false);
  const isRunning = engineStatus === "running" || engineStatus === "generating";

  useEffect(() => {
    if (!isRunning) return;
    const frameId = window.requestAnimationFrame(() => {
      messageEndRef.current?.scrollIntoView({ block: "end", behavior: "auto" });
    });
    return () => window.cancelAnimationFrame(frameId);
  }, [messages, isRunning]);

  useEffect(() => {
    if (!apiBase) return;
    axios.get(`${apiBase}/engine/status`).then((response) => {
      const adapter = response.data?.adapters?.[id.toLowerCase()];
      if (adapter) {
        setAdapterInfo({ name: adapter.name, path: adapter.path });
        setMountStatus("mounted");
        setMountMessage(response.data.status === "running" ? "Adapter mounted and loaded." : "Adapter validated and queued.");
      }
    }).catch(() => undefined);
  }, [apiBase, id]);

  useEffect(() => {
    if (wasRunningRef.current && engineStatus === "off") {
      setAdapterInfo(null);
      setAdapterError(null);
      setMountStatus("unmounted");
      setMountMessage("No adapter mounted.");
    }
    wasRunningRef.current = isRunning;
  }, [engineStatus, isRunning]);

  const mountAdapter = async () => {
    try {
      setAdapterError(null);
      if (!apiBase) throw new Error("Backend base URL is not configured");
      if (!modelPath) throw new Error("Select a local base model folder before mounting an adapter");
      const selected = await open({
        multiple: false,
        directory: true,
        title: `Mount adapter for ${id}`,
      });
      if (selected && typeof selected === "string") {
        const res = await axios.post(`${apiBase}/adapter/mount`, {
          adapter_path: selected,
          slot_id: id.toLowerCase(),
          base_model_path: modelPath,
        });
        if (res.data.status === "success") {
          const folderName = res.data.adapter?.name || selected.split(/[\\/]/).filter(Boolean).pop() || selected;
          setAdapterInfo({ name: folderName, path: res.data.adapter?.path || selected });
          setMountStatus("mounted");
          setMountMessage(res.data.message || (isRunning ? "Adapter mounted and loaded." : "Adapter validated and queued."));
        }
      }
    } catch (errorValue: any) {
      const message = errorValue.response?.data?.message || errorValue.response?.data?.detail || errorValue.message;
      setAdapterError(message);
      setMountStatus("failed");
      setMountMessage(message);
    }
  };

  const unmountAdapter = async () => {
    try {
      setAdapterError(null);
      if (!apiBase) throw new Error("Backend base URL is not configured");
      await axios.post(`${apiBase}/adapter/unmount`, {
        slot_id: id.toLowerCase(),
      });
      setAdapterInfo(null);
      setMountStatus("unmounted");
      setMountMessage("No adapter mounted.");
    } catch (errorValue: any) {
      const message = errorValue.response?.data?.message || errorValue.response?.data?.detail || errorValue.message;
      setAdapterError(message);
      setMountStatus("failed");
      setMountMessage(message);
    }
  };

  const statusStyle = mountStatus === "mounted"
    ? { dot: "bg-secondary active-glow", text: "text-secondary", panel: "bg-secondary/10 border-secondary text-secondary", label: "MOUNTED" }
    : mountStatus === "failed"
      ? { dot: "bg-error", text: "text-error", panel: "bg-error/10 border-error text-red-200", label: "MOUNT FAILED" }
      : { dot: "bg-amber-400", text: "text-amber-300", panel: "bg-amber-400/10 border-amber-400 text-amber-200", label: "NO ADAPTER" };
  const controlsDisabled = engineStatus === "booting" || engineStatus === "generating";

  return (
    <div className="flex-1 flex flex-col bg-cyber-black overflow-hidden border-r border-white/5 last:border-r-0">
      <div className="h-11 border-b border-white/10 flex justify-between items-center px-4 bg-black/40 shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <div className={`w-2 h-2 ${statusStyle.dot}`} />
          <span className={`text-[11px] font-bold uppercase truncate ${statusStyle.text}`}>
            Stream_{id} <span className="ml-2">[{mountStatus === "mounted" && adapterInfo ? adapterInfo.name : statusStyle.label}]</span>
          </span>
        </div>
        <div className="flex gap-2">
          <button onClick={mountAdapter} disabled={controlsDisabled || !modelPath} className={adapterInfo ? "btn-tool" : "btn-primary"}>
            <FolderOpen size={13} /> {adapterInfo ? "Remount" : "Mount"}
          </button>
          <button onClick={unmountAdapter} disabled={controlsDisabled || (!adapterInfo && mountStatus !== "failed")} className="btn-danger">
            <Unplug size={13} /> Unmount
          </button>
        </div>
      </div>
      <div className="flex-1 p-6 overflow-y-auto space-y-4 custom-scrollbar">
        {!isRunning ? (
          <div className="h-full flex flex-col items-center justify-center gap-4 px-6">
            <div className="flex flex-col items-center gap-3 text-error opacity-60">
              <Activity size={40} className="animate-pulse" />
              <span className="text-[10px] tracking-[0.8em] font-black">ENGINE_OFFLINE</span>
            </div>
            <div className={`w-full max-w-xl p-3 border-l-2 text-[11px] font-bold ${statusStyle.panel}`}>
              {mountMessage}
              {adapterInfo && <div className="mt-1 font-mono opacity-80 break-all">{adapterInfo.path}</div>}
            </div>
            {adapterError && <div className="w-full max-w-xl p-3 bg-error/10 border border-error/40 text-red-200 text-[11px] font-bold">{adapterError}</div>}
          </div>
        ) : (
          <>
            <div className={`p-3 border-l-2 text-[11px] font-bold mb-6 ${statusStyle.panel}`}>
              {mountMessage}
              {adapterInfo && <div className="mt-1 font-mono opacity-80 break-all">{adapterInfo.path}</div>}
            </div>
            {adapterError && <div className="p-3 bg-error/10 border border-error/40 text-red-200 text-[11px] font-bold">{adapterError}</div>}
            {messages.map((msg, index) => (
              <div key={index} className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                <span className="text-[8px] opacity-30 mb-1 uppercase font-black">{msg.role}</span>
                <div className={`max-w-[90%] p-3 text-xs border leading-relaxed ${msg.role === "user" ? "bg-neon-green/5 border-neon-green/20 text-neon-green" : "bg-white/5 border-white/10 text-white/80"}`}>
                  {msg.content || (msg.role === "assistant" && <span className="animate-pulse">...</span>)}
                </div>
              </div>
            ))}
            <div ref={messageEndRef} className="h-px" />
          </>
        )}
      </div>
    </div>
  );
}
