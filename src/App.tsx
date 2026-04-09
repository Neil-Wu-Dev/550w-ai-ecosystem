import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { open } from "@tauri-apps/plugin-dialog";
import { motion, AnimatePresence } from "framer-motion";
import {
  Cpu, Activity, Send, FileText,
  Play, Box, Columns, Square, ChevronRight, FolderOpen, Paperclip,
  CirclePower, XOctagon, AlertCircle, XCircle, Zap, StopCircle, RefreshCw, CheckCircle2
} from "lucide-react";

// 严格对齐后端 main.py 的 prefix
const API_BASE = "http://127.0.0.1:8000/api/v1/workflow";

// --- 类型定义 ---
interface ModelInfo {
  name: string;
  local_path: string;
  architecture: string;
  precision: string;
  trainable_layers: string[];
}

interface StaticOptions {
  strategies: string[];
  modes: string[];
}

interface AdapterData {
  name: string;
  path: string;
}

type EngineStatus = "off" | "booting" | "running";

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatHistory {
  alpha: ChatMessage[];
  beta: ChatMessage[];
}

// 进度字典定义
interface TrainingProgress {
  status: string;
  message: string;
  percentage: number;
  epoch?: number;
  step?: number;
}

export default function App() {
  const [options, setOptions] = useState<StaticOptions>({ strategies: [], modes: [] });
  const [selectedStrategy, setSelectedStrategy] = useState<string>("");
  const [selectedMode, setSelectedMode] = useState<string>("");
  const [modelPath, setModelPath] = useState<string>("");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [corpusPath, setCorpusPath] = useState<string>("");
  const [selectedLayers, setSelectedLayers] = useState<string[]>([]);
  
  // --- 新增状态：训练轮数 ---
  const [epochs, setEpochs] = useState<number>(1);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentProgress, setCurrentProgress] = useState<TrainingProgress | null>(null);
  const [showSuccessHint, setShowSuccessHint] = useState<boolean>(false);
  
  const [trainingTime, setTrainingTime] = useState<number>(0);
  const [engineStatus, setEngineStatus] = useState<EngineStatus>("off");
  const [isDualMode, setIsDualMode] = useState<boolean>(false);
  const [prompt, setPrompt] = useState<string>("");
  const [chatHistory, setChatHistory] = useState<ChatHistory>({ alpha: [], beta: [] });
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const trainingAbortRef = useRef<AbortController | null>(null);

  // 初始化选项
  useEffect(() => {
    axios.get<StaticOptions>(`${API_BASE}/init-options`)
      .then((res) => {
        setOptions(res.data);
        if (res.data.modes.length > 0) setSelectedMode(res.data.modes[0]);
        if (res.data.strategies.length > 0) setSelectedStrategy(res.data.strategies[0]);
      })
      .catch((e: any) => {
        setError(`INIT_FAILURE: ${e.response?.data?.detail || e.message}`);
      });
  }, []);

  useEffect(() => {
    let timer: number;
    if (isTraining) {
      timer = window.setInterval(() => setTrainingTime(t => t + 1), 1000);
    } else {
      setTrainingTime(0);
      timer = 0;
    }
    return () => clearInterval(timer);
  }, [isTraining]);

  const pickModelDir = async () => {
    try {
      const selected = await open({ directory: true, multiple: false, title: "LOAD_BASE_MODEL" });
      if (selected && typeof selected === 'string') {
        setModelPath(selected);
        const res = await axios.post<ModelInfo>(`${API_BASE}/select-model`, { local_path: selected });
        setModelInfo(res.data);
        setSelectedLayers([]);
      }
    } catch (e: any) { setError(e.response?.data?.detail || e.toString()); }
  };

  const pickCorpusFile = async () => {
    try {
      const selected = await open({
        directory: false, multiple: false,
        filters: [{ name: 'PDF', extensions: ['pdf'] }],
        title: "LOAD_CORPUS_SOURCE"
      });
      if (selected && typeof selected === 'string') setCorpusPath(selected);
    } catch (e: any) { setError(e.toString()); }
  };

  const toggleLayer = (layer: string) => {
    setSelectedLayers(prev => prev.includes(layer) ? prev.filter(l => l !== layer) : [...prev, layer]);
  };

  const handleStartTrain = async () => {
    if (!modelInfo || !corpusPath || selectedLayers.length === 0 || !selectedStrategy || !selectedMode) return;
    setIsTraining(true);
    setError(null);
    setShowSuccessHint(false);
    setCurrentProgress({ status: "initializing", message: "PREPARING_UPLINK", percentage: 0 });
    trainingAbortRef.current = new AbortController();

    try {
      await axios.post(`${API_BASE}/process-dataset`, null, {
          params: { file_path: corpusPath, chunk_size: 512, strategy: selectedStrategy }
      });

      const response = await fetch(`${API_BASE}/start-train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: trainingAbortRef.current.signal,
        body: JSON.stringify({
          target_dir: "./exports",
          layers: selectedLayers,
          mode: selectedMode,
          name: `adcust_${selectedStrategy}_${Date.now()}`,
          epochs: epochs // <--- 关键修改：透传手动输入的轮数
        })
      });

      if (!response.body) throw new Error("STREAM_INIT_FAILED");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data: TrainingProgress = JSON.parse(line);
            if (data.status === "error") throw new Error(data.message);
            
            // 实时更新进度状态
            setCurrentProgress(data);
            
            // 检查是否完成
            if (data.status === "completed") {
                setShowSuccessHint(true);
                setTimeout(() => setShowSuccessHint(false), 5000);
            }
          } catch(e) { /* 忽略不完整的 JSON 行 */ }
        }
      }
    } catch (e: any) { 
      if (e.name !== 'AbortError') setError(e.toString()); 
    } finally { 
      setIsTraining(false); 
    }
  };

  const handleStopTrain = async () => {
    try {
      if (trainingAbortRef.current) trainingAbortRef.current.abort();
      await axios.post(`${API_BASE}/stop-train`, { mode: "sequential" });
      setIsTraining(false);
      setCurrentProgress(prev => prev ? { ...prev, status: "aborted", message: "USER_TERMINATED" } : null);
    } catch (e: any) { setError(`STOP_FAILED: ${e.message}`); }
  };

  const controlEngine = async (action: 'boot' | 'shutdown') => {
    try {
      setError(null);
      if (action === 'boot') {
        setEngineStatus("booting");
        await axios.post(`${API_BASE}/engine/boot`);
        setEngineStatus("running");
      } else {
        if (abortControllerRef.current) abortControllerRef.current.abort();
        await axios.post(`${API_BASE}/engine/shutdown`);
        setEngineStatus("off");
        setChatHistory({ alpha: [], beta: [] }); 
      }
    } catch (e: any) {
      setError(e.response?.data?.detail || e.toString());
      setEngineStatus("off");
    }
  };

  const handleSendMessage = async () => {
    if (!prompt.trim() || engineStatus !== 'running') return;
    const userQuery = prompt;
    setPrompt(""); 
    setChatHistory(prev => ({
      alpha: [...prev.alpha, { role: 'user', content: userQuery }, { role: 'assistant', content: "" }],
      beta: isDualMode ? [...prev.beta, { role: 'user', content: userQuery }, { role: 'assistant', content: "" }] : prev.beta
    }));
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          prompt: userQuery,
          targets: isDualMode ? [{ id: "alpha" }, { id: "beta" }] : [{ id: "alpha" }],
          strategy: "interleaved",
          max_tokens: 512
        })
      });
      if (!response.body) throw new Error("Stream_Unavailable");
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
          try {
            const data = JSON.parse(trimmedLine);
            const { source, token, is_final } = data;
            if (is_final) continue;
            setChatHistory(prev => {
              const targetKey = source === "beta" ? "beta" : "alpha";
              const updatedList = [...prev[targetKey]];
              const lastIndex = updatedList.length - 1;
              if (lastIndex >= 0) {
                updatedList[lastIndex] = { ...updatedList[lastIndex], content: updatedList[lastIndex].content + token };
              }
              return { ...prev, [targetKey]: updatedList };
            });
          } catch (e) { console.warn("JSON_ERR", trimmedLine); }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') setError(`INFERENCE_ERROR: ${e.message}`);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-cyber-black text-cyber-green overflow-hidden relative font-mono text-sm">
      <div className="crt-overlay" />
      <header className="h-14 border-b border-cyber-green/20 flex items-center justify-between px-6 bg-surface-dark z-10">
        <div className="flex items-center gap-4">
          <div className="p-2 bg-neon-green/10 rounded-sm"><Box className="text-neon-green" size={20} /></div>
          <div>
            <h1 className="font-black tracking-widest text-sm leading-none">ADCUST_SYSTEM_Prototype</h1>
            <p className="text-[9px] opacity-40 mt-1 uppercase">Adapter Customizer </p>
          </div>
        </div>
        <div className="flex items-center gap-6 text-[10px]">
          <div className="flex flex-col items-end">
            <span className="opacity-40">MODEL_IDENTIFIED</span>
            <span className={modelInfo ? "text-neon-green" : "text-red-500 font-bold"}>
              {modelInfo ? modelInfo.name : "NULL_ASSET"}
            </span>
          </div>
          <div className="h-8 w-px bg-cyber-green/10" />
          <div className="flex items-center gap-3 bg-black/40 px-4 py-2 border border-cyber-green/10">
            <div className={`w-2 h-2 rounded-full ${engineStatus === 'running' ? 'bg-neon-green animate-pulse' : engineStatus === 'booting' ? 'bg-amber-500 animate-bounce' : 'bg-red-500'}`} />
            <span className="uppercase tracking-widest font-bold">CORE: {engineStatus}</span>
          </div>
        </div>
      </header>

      <main className="flex-1 flex overflow-hidden z-10">
        <aside className="w-96 border-r border-cyber-green/10 bg-surface-dark p-6 flex flex-col gap-6 overflow-y-auto custom-scrollbar">
          {/* 01_MODEL_CONFIG */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold opacity-40 flex items-center gap-2"><Cpu size={14}/> 01_MODEL_CONFIG</h3>
            <button onClick={pickModelDir} className="w-full flex items-center justify-between bg-black border border-cyber-green/20 p-3 hover:border-neon-green transition-all">
              <span className="text-[10px] truncate pr-4 opacity-70 font-bold">{modelPath || "OPEN_BASE_MODEL_DIRECTORY"}</span>
              <FolderOpen size={16} />
            </button>
            {modelInfo && (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4 p-3 bg-black/40 border border-cyber-green/5 text-[10px]">
                {/* 新增组件：训练次数选择 */}
                <div className="border-b border-cyber-green/10 pb-3">
                  <p className="opacity-40 uppercase mb-2 flex items-center gap-2 text-neon-green"><RefreshCw size={10}/> Iteration_Epochs:</p>
                  <div className="flex items-center gap-3">
                    <input 
                      type="number" 
                      min="1" 
                      max="100"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value) || 1)}
                      className="flex-1 bg-black border border-cyber-green/30 p-2 text-neon-green outline-none focus:border-neon-green transition-all font-black text-center"
                    />
                    <div className="text-[9px] opacity-30 flex flex-col uppercase font-bold">
                        <span>Min: 01</span>
                        <span>Max: 500</span>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="opacity-40 uppercase mb-2">Target Layers:</p>
                  <div className="grid grid-cols-2 gap-2">
                    {modelInfo.trainable_layers.map(l => (
                      <button key={l} onClick={() => toggleLayer(l)} className={`py-1.5 px-2 border transition-all truncate text-left ${selectedLayers.includes(l) ? 'bg-neon-green/20 border-neon-green text-neon-green' : 'border-white/5 opacity-40'}`}>
                        {l}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="opacity-40 mb-2 uppercase">Training Mode:</p>
                  <div className="flex flex-wrap gap-2">
                    {options.modes.map(m => (
                      <button key={m} onClick={() => setSelectedMode(m)} className={`flex-1 min-w-[80px] py-2 border uppercase text-[9px] transition-all ${selectedMode === m ? 'bg-neon-green text-black border-neon-green font-black shadow-[0_0_10px_#39ff14]' : 'border-white/10 opacity-40'}`}>
                        {m}
                      </button>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}
          </section>

          {/* 02_DATA_SOURCE */}
          <section className="space-y-4">
            <h3 className="text-xs font-bold opacity-40 flex items-center gap-2"><FileText size={14}/> 02_DATA_SOURCE</h3>
            <div className="p-3 bg-black/40 border border-cyber-green/5 space-y-3">
               <p className="text-[9px] opacity-40 uppercase flex items-center gap-2"><Zap size={10}/> Optimization_Strategy:</p>
               <div className="flex flex-col gap-2">
                 {options.strategies.map(s => (
                   <button key={s} onClick={() => setSelectedStrategy(s)} className={`w-full py-2 px-3 border text-left text-[10px] uppercase transition-all flex justify-between items-center ${selectedStrategy === s ? 'bg-neon-green/10 border-neon-green text-neon-green font-black shadow-[0_0_10px_rgba(57,255,20,0.1)]' : 'border-white/5 opacity-40 hover:opacity-100'}`}>
                     {s}
                     {selectedStrategy === s && <div className="w-1.5 h-1.5 bg-neon-green rounded-full shadow-[0_0_5px_#39ff14]"/>}
                   </button>
                 ))}
               </div>
            </div>
            <button onClick={pickCorpusFile} className={`w-full border-2 border-dashed py-8 text-[10px] transition-all flex flex-col items-center gap-3 ${corpusPath ? 'border-neon-green bg-neon-green/5 text-neon-green font-black' : 'border-cyber-green/10 opacity-60'}`}>
              <Paperclip size={20} className={corpusPath ? "animate-bounce text-neon-green" : ""} />
              <span className="uppercase tracking-widest text-center px-4 truncate w-full">{corpusPath ? corpusPath.split(/[\\/]/).pop() : 'LOAD_PDF_CORPUS'}</span>
            </button>
          </section>

          {/* 增强型训练进展组件 */}
          <div className="mt-auto flex flex-col gap-2">
            <AnimatePresence>
                {(isTraining || currentProgress) && (
                    <motion.div 
                        initial={{ opacity: 0, height: 0 }} 
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="bg-black/80 border border-cyber-green/20 p-3 text-[10px] space-y-2 overflow-hidden"
                    >
                        <div className="flex justify-between items-center uppercase font-black">
                            <span className="text-neon-green">{currentProgress?.status || 'INIT'}</span>
                            <span className="text-white/40">{currentProgress?.percentage || 0}%</span>
                        </div>
                        {/* 进度条 */}
                        <div className="h-1 bg-white/5 w-full relative overflow-hidden">
                            <motion.div 
                                className="h-full bg-neon-green shadow-[0_0_8px_#39ff14]" 
                                animate={{ width: `${currentProgress?.percentage || 0}%` }}
                            />
                        </div>
                        {/* 详细指标 */}
                        <div className="flex gap-4 text-[9px] opacity-60 font-bold uppercase border-t border-white/5 pt-2">
                            <div className="flex flex-col">
                                <span>Epoch</span>
                                <span className="text-neon-green">{currentProgress?.epoch || '--'}/{epochs}</span>
                            </div>
                            <div className="flex flex-col">
                                <span>Batch_Step</span>
                                <span className="text-neon-green">{currentProgress?.step || '--'}</span>
                            </div>
                            <div className="flex flex-col flex-1 truncate">
                                <span>Action</span>
                                <span className="truncate italic">{currentProgress?.message || 'PENDING...'}</span>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* 训练成功提示 */}
            <AnimatePresence>
                {showSuccessHint && (
                    <motion.div 
                        initial={{ scale: 0.9, opacity: 0 }} 
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        className="bg-neon-green/10 border border-neon-green p-3 flex items-center gap-3"
                    >
                        <CheckCircle2 className="text-neon-green" size={18}/>
                        <div className="flex-1">
                            <p className="text-[10px] font-black uppercase text-neon-green">Training_Completed</p>
                            <p className="text-[9px] opacity-70">Artifacts exported to system local storage.</p>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {!isTraining ? (
                <button disabled={!modelInfo || !corpusPath || selectedLayers.length === 0 || !selectedStrategy} onClick={handleStartTrain} className={`py-5 font-black text-xs flex items-center justify-center gap-3 transition-all tracking-[0.3em] uppercase border-2 ${(!modelInfo || !corpusPath || selectedLayers.length === 0 || !selectedStrategy) ? 'opacity-20 grayscale cursor-not-allowed border-white/10' : 'bg-neon-green text-black border-neon-green hover:shadow-[0_0_25px_#39ff14]'}`}><Play size={16}/> START_TRAIN</button>
            ) : (
                <button onClick={handleStopTrain} className="py-5 font-black text-xs flex items-center justify-center gap-3 transition-all tracking-[0.3em] uppercase border-2 bg-red-600 text-white border-red-500 hover:bg-red-700 shadow-[0_0_15px_rgba(255,0,0,0.4)]"><StopCircle size={16}/> STOP_TRAIN</button>
            )}
          </div>
        </aside>

        {/* Inference View */}
        <section className="flex-1 flex flex-col bg-cyber-black relative border-l border-cyber-green/10">
          <div className="h-14 border-b border-cyber-green/10 flex items-center justify-between px-6 bg-surface-light/30">
            <div className="flex gap-8">
              <button onClick={() => setIsDualMode(false)} className={`text-[10px] flex items-center gap-2 tracking-widest ${!isDualMode ? 'text-neon-green font-bold' : 'opacity-20 hover:opacity-100'}`}><Square size={12}/> SINGLE_VIEW</button>
              <button onClick={() => setIsDualMode(true)} className={`text-[10px] flex items-center gap-2 tracking-widest ${isDualMode ? 'text-neon-green font-bold' : 'opacity-20 hover:opacity-100'}`}><Columns size={12}/> DUAL_SYNC</button>
            </div>
            <AnimatePresence mode="wait">
              {engineStatus === "off" ? (
                <motion.button key="boot" onClick={() => controlEngine('boot')} className="flex items-center gap-2 bg-neon-green/10 border border-neon-green/40 px-5 py-1.5 text-[10px] text-neon-green hover:bg-neon-green hover:text-black font-black"><CirclePower size={14} /> IGNITE_ENGINE</motion.button>
              ) : (
                <motion.button key="shutdown" onClick={() => controlEngine('shutdown')} disabled={engineStatus === "booting"} className="flex items-center gap-2 bg-red-950 border border-red-500/50 px-5 py-1.5 text-[10px] text-red-500 hover:bg-red-500 hover:text-white font-black"><XOctagon size={14} /> SHUTDOWN</motion.button>
              )}
            </AnimatePresence>
          </div>

          <div className="flex-1 flex gap-px bg-cyber-green/5 overflow-hidden">
            <ChatPanel id="ALPHA" isRunning={engineStatus === "running"} messages={chatHistory.alpha} />
            {isDualMode && <ChatPanel id="BETA" isRunning={engineStatus === "running"} messages={chatHistory.beta} />}
          </div>

          <div className="p-6 bg-surface-dark border-t border-cyber-green/10">
            <div className={`relative flex items-center p-1 border transition-all ${engineStatus === 'running' ? 'border-neon-green/40 bg-black' : 'border-white/5 opacity-20'}`}>
              <div className="pl-4 pr-2 text-neon-green opacity-40"><ChevronRight size={18} /></div>
              <input disabled={engineStatus !== 'running'} className="flex-1 bg-transparent border-none outline-none text-neon-green py-4 text-sm font-bold" placeholder={engineStatus === 'running' ? "COMMAND_UPLINK..." : "ENGINE_OFFLINE"} value={prompt} onChange={e => setPrompt(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSendMessage()} />
              <button disabled={engineStatus !== 'running' || !prompt.trim()} onClick={handleSendMessage} className="p-4 text-neon-green hover:scale-110 disabled:opacity-0"><Send size={20}/></button>
            </div>
          </div>
          
          <AnimatePresence>
            {error && (
              <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 20 }} className="absolute bottom-32 left-6 right-6 z-50 p-4 bg-red-950 border-l-4 border-red-500 flex items-start gap-4 shadow-2xl">
                <AlertCircle className="text-red-500 shrink-0 mt-0.5" size={18} />
                <div className="flex-1 overflow-hidden">
                  <h4 className="text-[10px] font-black text-red-500 uppercase mb-1">Critical_Error</h4>
                  <p className="text-xs text-red-200 font-bold truncate">{error}</p>
                </div>
                <button onClick={() => setError(null)} className="text-red-500 hover:text-white"><XCircle size={18} /></button>
              </motion.div>
            )}
          </AnimatePresence>
        </section>
      </main>

      <footer className="h-8 border-t border-cyber-green/20 bg-cyber-black flex items-center justify-between px-6 text-[9px] font-black opacity-40">
        <div className="flex gap-8"><span>OS: TAURI_V2</span><span>UI: REACT_18</span></div>
        <div className="flex items-center gap-2"><div className={`w-1.5 h-1.5 rounded-full ${isTraining ? 'bg-amber-500 animate-ping' : 'bg-neon-green'}`} /><span>{isTraining ? `TRAINING_IN_PROGRESS: ${trainingTime}s` : 'SYSTEM_READY'}</span></div>
      </footer>
    </div>
  );
}

function ChatPanel({ id, isRunning, messages }: { id: string, isRunning: boolean, messages: ChatMessage[] }) {
  const [adapterInfo, setAdapterInfo] = useState<AdapterData | null>(null);
  
  const mountAdapter = async () => {
    try {
      const selected = await open({
        multiple: false,
        directory: true, 
        title: `MOUNT_ADAPTER_DIRECTORY_${id}`
      });

      if (selected && typeof selected === 'string') {
        const res = await axios.post(`${API_BASE}/adapter/mount`, {
            adapter_path: selected, 
            slot_id: id.toLowerCase() 
        });
        
        if (res.data.status === "success") {
          const folderName = selected.split(/[\\/]/).filter(Boolean).pop() || "ADAPTER";
          setAdapterInfo({ name: folderName, path: selected });
        }
      }
    } catch (e: any) { 
        const errMsg = e.response?.data?.detail || e.message;
        alert(`MOUNT_ERROR: ${errMsg}`); 
    }
  };

  return (
    <div className="flex-1 flex flex-col bg-cyber-black overflow-hidden border-r border-white/5 last:border-r-0">
      <div className="h-9 border-b border-white/5 flex justify-between items-center px-4 bg-black/40 shrink-0">
        <div className="flex items-center gap-2">
          <div className={`w-1.5 h-1.5 rounded-full ${isRunning ? 'bg-neon-green' : 'bg-red-500'}`} />
          <span className="text-[10px] font-black text-neon-green uppercase tracking-widest">
            Stream_{id} {adapterInfo && <span className="text-amber-500 ml-2">[{adapterInfo.name}]</span>}
          </span>
        </div>
        <button onClick={mountAdapter} className={`text-[8px] border px-3 py-1 font-black uppercase transition-all ${adapterInfo ? 'border-amber-500 text-amber-500' : 'border-cyber-green/30 text-cyber-green hover:bg-neon-green hover:text-black'}`}>
          {adapterInfo ? 'Remount' : 'Mount_Adapter'}
        </button>
      </div>
      <div className="flex-1 p-6 overflow-y-auto custom-scrollbar space-y-4">
        {!isRunning ? (
          <div className="h-full flex flex-col items-center justify-center opacity-10 grayscale gap-3">
            <Activity size={40} />
            <span className="text-[10px] tracking-[0.8em] font-black">NO_UPLINK</span>
          </div>
        ) : (
          <>
            <div className="p-3 bg-white/5 border-l-2 border-neon-green/40 text-neon-green/60 text-[10px] italic font-bold mb-6">
              Neural_Handshake_Established... Ready. {adapterInfo && `(Active: ${adapterInfo.name})`}
            </div>
            {messages.map((msg, i) => (
              <div key={i} className={`flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                <span className="text-[8px] opacity-30 mb-1 uppercase font-black">{msg.role}</span>
                <div className={`max-w-[90%] p-3 text-xs border ${msg.role === 'user' ? 'bg-neon-green/5 border-neon-green/20 text-neon-green' : 'bg-white/5 border-white/10 text-white/80'}`}>{msg.content}</div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}