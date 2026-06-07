import { useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import { AlertTriangle, CheckCircle2, ClipboardCopy, FileText, FolderDown, FolderOpen, Play, RefreshCw, Server, Square, Timer, WalletCards } from "lucide-react";

type PageProps = {
  apiBase: string;
  backendReady: boolean;
};

type Provider = {
  id: string;
  name: string;
  endpoint_summary: string;
  remote_workspace_path: string;
  hourly_rate_usd: number;
  is_active: boolean;
  last_heartbeat?: string;
};

type ProgressEvent = {
  job_id?: string;
  status: string;
  message: string;
  percentage?: number;
  elapsed_seconds?: number;
  estimated_cost_usd?: number;
  local_output_dir?: string;
  stage?: string;
  loss?: number;
  train_progress?: number;
  eta_seconds?: number;
  step?: number;
  total_steps?: number;
  error?: string;
  resources?: ResourceStatus[];
  download_progress?: number;
  downloaded_bytes?: number;
  total_bytes?: number;
  document_count?: number;
  raw_sample_count?: number;
  qa_pair_count?: number;
  qa_sample_count?: number;
  training_sample_count?: number;
  actual_steps?: number;
  final_loss?: number;
};

type ResourceStatus = {
  id: string;
  kind: string;
  name: string;
  status: string;
  size_bytes?: number;
  download_progress?: number;
  downloaded_bytes?: number;
  total_bytes?: number;
};

const TRAINING_DRAFT_KEY = "adcust.trainingDraft.v1";
const TRAINING_RUNTIME_KEY = "adcust.trainingRuntime.v1";
const ACTIVE_COMPUTE_KEY = "adcust.activeComputeNodeId";

type TrainingDraft = {
  providerId?: string;
  datasetPath?: string;
  localBaseModelPath?: string;
  baseModel?: string;
  localOutputRoot?: string;
  epochs?: string;
  learningRate?: string;
  maxSeqLength?: string;
  loraRank?: string;
  loraAlpha?: string;
  loraDropout?: string;
  targetModules?: string;
  useQlora?: boolean;
  cleanupRemote?: boolean;
};

const loadDraft = (): TrainingDraft => {
  try {
    return JSON.parse(localStorage.getItem(TRAINING_DRAFT_KEY) || "{}");
  } catch {
    return {};
  }
};

type TrainingRuntime = {
  latest?: ProgressEvent | null;
  logs?: string[];
  lossPoints?: { step: number; loss: number }[];
  error?: string | null;
};

const loadRuntime = (): TrainingRuntime => {
  try {
    return JSON.parse(localStorage.getItem(TRAINING_RUNTIME_KEY) || "{}");
  } catch {
    return {};
  }
};

const saveRuntime = (runtime: TrainingRuntime) => {
  localStorage.setItem(TRAINING_RUNTIME_KEY, JSON.stringify(runtime));
};

const PIPELINE_STEPS = [
  { label: "Connect to SSH node", start: 1, complete: 7 },
  { label: "Check and prepare environment", start: 8, complete: 15 },
  { label: "Prepare and upload training assets", start: 16, complete: 33 },
  { label: "Load training configuration and dataset", start: 35, complete: 40 },
  { label: "Download and verify exact base model", start: 41, complete: 58 },
  { label: "Run raw-knowledge LoRA / QLoRA training", start: 59, complete: 85 },
  { label: "Save adapter artifacts", start: 86, complete: 89 },
  { label: "Download adapter to local storage", start: 90, complete: 95 },
  { label: "Clean remote job workspace", start: 96, complete: 99, optional: true },
  { label: "Finish training job", start: 100, complete: 100 },
];

export default function Training({ apiBase, backendReady }: PageProps) {
  const draft = useMemo(loadDraft, []);
  const runtime = useMemo(loadRuntime, []);
  const [providers, setProviders] = useState<Provider[]>([]);
  const [providerId, setProviderId] = useState(() => localStorage.getItem(ACTIVE_COMPUTE_KEY) || draft.providerId || "");
  const [datasetPath, setDatasetPath] = useState(draft.datasetPath || "");
  const [localBaseModelPath, setLocalBaseModelPath] = useState(draft.localBaseModelPath || "");
  const [baseModel, setBaseModel] = useState(draft.baseModel || "");
  const [adapterName, setAdapterName] = useState("");
  const [localOutputRoot, setLocalOutputRoot] = useState(draft.localOutputRoot || "");
  const [epochs, setEpochs] = useState(draft.epochs || "");
  const [learningRate, setLearningRate] = useState(draft.learningRate || "");
  const [maxSeqLength, setMaxSeqLength] = useState(draft.maxSeqLength || "");
  const [loraRank, setLoraRank] = useState(draft.loraRank || "");
  const [loraAlpha, setLoraAlpha] = useState(draft.loraAlpha || "");
  const [loraDropout, setLoraDropout] = useState(draft.loraDropout || "");
  const [targetModules, setTargetModules] = useState(draft.targetModules || "");
  const [useQlora, setUseQlora] = useState(draft.useQlora ?? true);
  const [cleanupRemote, setCleanupRemote] = useState(draft.cleanupRemote ?? false);
  const [isTraining, setIsTraining] = useState(false);
  const [latest, setLatest] = useState<ProgressEvent | null>(runtime.latest || null);
  const [logs, setLogs] = useState<string[]>(runtime.logs || []);
  const [lossPoints, setLossPoints] = useState<{ step: number; loss: number }[]>(runtime.lossPoints || []);
  const [error, setError] = useState<string | null>(runtime.error || null);
  const abortRef = useRef<AbortController | null>(null);
  const activeJobIdRef = useRef<string | null>(null);
  const runtimeSnapshotRef = useRef(localStorage.getItem(TRAINING_RUNTIME_KEY) || "{}");

  const trainingApi = useMemo(() => (apiBase ? `${apiBase}/api/v1/training` : ""), [apiBase]);
  const computeApi = useMemo(() => (apiBase ? `${apiBase}/api/v1/compute` : ""), [apiBase]);
  const selectedProvider = useMemo(() => providers.find((p) => p.id === providerId), [providers, providerId]);
  const targetModuleList = targetModules.split(",").map((x) => x.trim()).filter(Boolean);
  const mergeProgressEvent = (previous: ProgressEvent | null, incoming: ProgressEvent): ProgressEvent => ({
    ...previous,
    ...incoming,
    percentage: Math.max(previous?.percentage || 0, incoming.percentage || 0),
    train_progress: incoming.train_progress ?? previous?.train_progress,
    step: incoming.step ?? previous?.step,
    total_steps: incoming.total_steps ?? previous?.total_steps,
    loss: incoming.loss ?? previous?.loss,
    eta_seconds: incoming.eta_seconds ?? previous?.eta_seconds,
    resources: incoming.resources?.length ? incoming.resources : previous?.resources,
  });

  const formatApiError = (value: unknown): string => {
    if (typeof value === "string") return value;
    if (Array.isArray(value)) return value.map(formatApiError).filter(Boolean).join("; ");
    if (value && typeof value === "object") {
      const item = value as Record<string, unknown>;
      if (typeof item.msg === "string") {
        const location = Array.isArray(item.loc) ? item.loc.join(".") : "";
        return location ? `${location}: ${item.msg}` : item.msg;
      }
      for (const key of ["message", "detail", "error", "reason"]) {
        if (item[key] !== undefined) {
          const formatted = formatApiError(item[key]);
          if (formatted) return formatted;
        }
      }
      try {
        return JSON.stringify(value);
      } catch {
        return String(value);
      }
    }
    return value == null ? "" : String(value);
  };

  const explainError = async (res: Response) => {
    const data = await res.json().catch(async () => await res.text().catch(() => ""));
    return formatApiError(data) || `${res.status} ${res.statusText}`;
  };

  const fetchProviders = async () => {
    if (!computeApi) return;
    const res = await fetch(`${computeApi}/`);
    if (res.ok) setProviders(await res.json());
  };

  const chooseDataset = async () => {
    const selected = await open({
      multiple: false,
      directory: false,
      title: "Select training dataset",
      filters: [{ name: "Dataset", extensions: ["json", "jsonl", "txt", "md", "pdf"] }],
    });
    if (selected && typeof selected === "string") setDatasetPath(selected);
  };

  const chooseLocalBaseModel = async () => {
    const selected = await open({
      multiple: false,
      directory: true,
      title: "Select local base model folder used for inference",
    });
    if (selected && typeof selected === "string") setLocalBaseModelPath(selected);
  };

  const chooseOutputRoot = async () => {
    const selected = await open({
      multiple: false,
      directory: true,
      title: "Select local adapter output folder",
    });
    if (selected && typeof selected === "string") setLocalOutputRoot(selected);
  };

  const startTraining = async () => {
    if (!trainingApi) return setError("Backend base URL is not configured");
    let runtimeLogs: string[] = [];
    let runtimeLossPoints: { step: number; loss: number }[] = [];
    setLogs([]);
    setLossPoints([]);
    setLatest(null);
    setError(null);
    saveRuntime({ latest: null, logs: [], lossPoints: [], error: null });
    setIsTraining(true);
    activeJobIdRef.current = null;
    abortRef.current = new AbortController();
    try {
      const res = await fetch(`${trainingApi}/remote/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          provider_id: providerId,
          dataset_path: datasetPath,
          local_base_model_path: localBaseModelPath,
          base_model_name_or_path: baseModel.trim(),
          adapter_name: adapterName.trim(),
          local_output_root: localOutputRoot,
          epochs: Number(epochs),
          learning_rate: Number(learningRate),
          max_seq_length: Number(maxSeqLength),
          lora_rank: Number(loraRank),
          lora_alpha: Number(loraAlpha),
          lora_dropout: Number(loraDropout),
          target_modules: targetModuleList,
          use_qlora: useQlora,
          cleanup_remote: cleanupRemote,
        }),
      });
      if (!res.ok) throw new Error(await explainError(res));
      if (!res.body) throw new Error("Training stream was not returned by the backend");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line) as ProgressEvent;
          if (event.job_id) activeJobIdRef.current = event.job_id;
          let mergedEvent = event;
          setLatest((previous) => {
            mergedEvent = mergeProgressEvent(previous, event);
            return mergedEvent;
          });
          runtimeLogs = [`[${new Date().toLocaleTimeString()}] ${event.status}: ${event.message}`, ...runtimeLogs].slice(0, 120);
          if (typeof event.loss === "number") {
            const point = { step: Number(event.step || runtimeLossPoints.length + 1), loss: event.loss };
            runtimeLossPoints = [
              ...runtimeLossPoints.filter((existing) => existing.step !== point.step),
              point,
            ].sort((left, right) => left.step - right.step).slice(-120);
            setLossPoints(runtimeLossPoints);
          }
          const eventError = event.status === "failed"
            ? formatApiError(event.error || event.message) || "Remote training failed"
            : null;
          if (eventError) setError(eventError);
          setLogs(runtimeLogs);
          saveRuntime({ latest: mergedEvent, logs: runtimeLogs, lossPoints: runtimeLossPoints, error: eventError });
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        const event = { status: "failed", message: err.message, error: err.message };
        runtimeLogs = [`[${new Date().toLocaleTimeString()}] failed: ${err.message}`, ...runtimeLogs].slice(0, 120);
        setLatest(event);
        setError(err.message);
        setLogs(runtimeLogs);
        saveRuntime({ latest: event, logs: runtimeLogs, lossPoints: runtimeLossPoints, error: err.message });
      }
    } finally {
      setIsTraining(false);
      abortRef.current = null;
    }
  };

  const stopTraining = async () => {
    const jobId = activeJobIdRef.current || latest?.job_id;
    if (jobId && trainingApi) {
      try {
        const res = await fetch(`${trainingApi}/remote/${jobId}/abort`, { method: "POST" });
        if (!res.ok) throw new Error(await explainError(res));
        const data = await res.json();
        setLatest(data);
        const nextLogs = [`[${new Date().toLocaleTimeString()}] aborted: ${data.message}`, ...logs].slice(0, 120);
        setLogs(nextLogs);
        saveRuntime({ latest: data, logs: nextLogs, lossPoints, error });
      } catch (err: any) {
        setError(err.message);
      }
    }
    abortRef.current?.abort();
    setIsTraining(false);
  };

  const refreshJobStatus = async () => {
    const jobId = activeJobIdRef.current || latest?.job_id;
    if (!jobId || !trainingApi) return;
    const res = await fetch(`${trainingApi}/remote/${jobId}`);
    if (res.ok) {
      const data = await res.json();
      const merged = mergeProgressEvent(latest, data);
      setLatest(merged);
      saveRuntime({ latest: merged, logs, lossPoints, error });
    }
  };

  const copyLogs = async () => {
    const content = logs.join("\n");
    if (!content) return;
    await navigator.clipboard.writeText(content);
  };

  useEffect(() => {
    fetchProviders().catch(() => null);
    const timer = window.setInterval(() => fetchProviders().catch(() => null), 8000);
    return () => window.clearInterval(timer);
  }, [computeApi]);

  useEffect(() => {
    const activeCompute = localStorage.getItem(ACTIVE_COMPUTE_KEY);
    if (activeCompute && activeCompute !== providerId) setProviderId(activeCompute);
  }, [providers]);

  useEffect(() => {
    const nextDraft: TrainingDraft = {
      providerId,
      datasetPath,
      localBaseModelPath,
      baseModel,
      localOutputRoot,
      epochs,
      learningRate,
      maxSeqLength,
      loraRank,
      loraAlpha,
      loraDropout,
      targetModules,
      useQlora,
      cleanupRemote,
    };
    localStorage.setItem(TRAINING_DRAFT_KEY, JSON.stringify(nextDraft));
    if (providerId) localStorage.setItem(ACTIVE_COMPUTE_KEY, providerId);
  }, [providerId, datasetPath, localBaseModelPath, baseModel, localOutputRoot, epochs, learningRate, maxSeqLength, loraRank, loraAlpha, loraDropout, targetModules, useQlora, cleanupRemote]);

  useEffect(() => {
    const nextRuntime = { latest, logs, lossPoints, error };
    runtimeSnapshotRef.current = JSON.stringify(nextRuntime);
    saveRuntime(nextRuntime);
  }, [latest, logs, lossPoints, error]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      if (isTraining) return;
      const raw = localStorage.getItem(TRAINING_RUNTIME_KEY) || "{}";
      if (raw === runtimeSnapshotRef.current) return;
      try {
        const next = JSON.parse(raw) as TrainingRuntime;
        runtimeSnapshotRef.current = raw;
        setLatest(next.latest || null);
        setLogs(next.logs || []);
        setLossPoints(next.lossPoints || []);
        setError(next.error || null);
        if (next.latest?.job_id) activeJobIdRef.current = next.latest.job_id;
      } catch {
        return;
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [isTraining]);

  useEffect(() => {
    const active = latest?.job_id && !["completed", "failed", "aborted"].includes(latest.status);
    if (!active || isTraining) return;
    const timer = window.setInterval(() => refreshJobStatus().catch(() => null), 3000);
    return () => window.clearInterval(timer);
  }, [latest?.job_id, latest?.status, isTraining, trainingApi]);

  const workflow = [
    { label: "Backend connected", ok: backendReady },
    { label: "Compute node selected", ok: Boolean(providerId) },
    { label: "SSH node verified", ok: Boolean(selectedProvider?.is_active) },
    { label: "Dataset selected", ok: Boolean(datasetPath) },
    { label: "Local base model selected", ok: Boolean(localBaseModelPath) },
    { label: "Remote base model set", ok: Boolean(baseModel.trim()) },
    { label: "Adapter name set", ok: Boolean(adapterName.trim()) },
    { label: "Local output folder selected", ok: Boolean(localOutputRoot) },
    { label: "LoRA parameters completed", ok: Boolean(epochs && learningRate && maxSeqLength && loraRank && loraAlpha && loraDropout && targetModuleList.length > 0) },
  ];

  const canStart = workflow.every((step) => step.ok) && !isTraining;
  const progress = latest?.percentage ?? 0;
  const resources = latest?.resources || [];
  const etaLabel = formatDuration(latest?.eta_seconds);
  const trainingProgress = typeof latest?.train_progress === "number" ? `${latest.train_progress.toFixed(1)}%` : "waiting";

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(360px,440px)_minmax(0,1fr)] items-start gap-5 pb-6">
      <section className="border border-white/10 bg-surface-container-low p-5 min-w-0">
        <div className="mb-5">
          <h1 className="text-xl font-bold">Remote Adapter Training</h1>
          <p className="text-xs text-slate-500 mt-1">AdCust prepares dependencies, converts PDF text, downloads the HuggingFace base model on the server, verifies the model hash, then trains.</p>
        </div>

        <div className="space-y-4">
          <PanelTitle icon={<Server size={15} />} text="Compute Node" />
          <div className="p-3 border border-error/50 bg-error/10 text-red-100 text-[11px] font-bold leading-relaxed">
            You must manually start the server on your provider website before connecting. AdCust only verifies and uses SSH. Disconnecting AdCust does not stop, destroy, or stop billing for the server.
          </div>
          <select value={providerId} onChange={(e) => setProviderId(e.target.value)} className="w-full h-10 bg-black/40 border border-white/10 px-3 text-xs outline-none focus:border-primary">
            <option value="">Select a verified SSH node</option>
            {providers.map((p) => <option key={p.id} value={p.id}>{p.name} - {p.endpoint_summary}</option>)}
          </select>
          {selectedProvider && (
            <div className="text-[11px] text-slate-500 bg-black/20 border border-white/5 p-3">
              SSH status: {selectedProvider.is_active ? "verified" : "not verified"}<br />
              Remote workspace: {selectedProvider.remote_workspace_path}<br />
              Cost rate: ${Number(selectedProvider.hourly_rate_usd).toFixed(4)} / hour
            </div>
          )}

          <PanelTitle icon={<FileText size={15} />} text="Assets And Destination" />
          <button onClick={chooseDataset} className="file-picker-button"><FileText size={15} /> {datasetPath || "Select dataset file (JSON / JSONL / TXT / MD / PDF)"}</button>
          <button onClick={chooseLocalBaseModel} className="file-picker-button"><FolderOpen size={15} /> {localBaseModelPath || "Select local base model folder for strict binding"}</button>
          <Field label="HuggingFace Model ID or Existing Remote Model Path" value={baseModel} onChange={setBaseModel} />
          <div className="text-[11px] text-slate-500 border border-white/10 bg-black/20 p-3 leading-relaxed">
            If this value is a HuggingFace model ID, AdCust downloads it into the remote workspace automatically. Training starts only after the downloaded remote model manifest matches the selected local model.
          </div>
          <Field label="Adapter Name" value={adapterName} onChange={setAdapterName} />
          <button onClick={chooseOutputRoot} className="file-picker-button"><FolderOpen size={15} /> {localOutputRoot || "Select local adapter output folder"}</button>

          <PanelTitle icon={<Timer size={15} />} text="LoRA / QLoRA Parameters" />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Epochs" type="number" value={epochs} onChange={setEpochs} />
            <Field label="Max Sequence" type="number" value={maxSeqLength} onChange={setMaxSeqLength} />
            <Field label="Learning Rate" type="number" value={learningRate} onChange={setLearningRate} />
            <Field label="LoRA Rank" type="number" value={loraRank} onChange={setLoraRank} />
            <Field label="LoRA Alpha" type="number" value={loraAlpha} onChange={setLoraAlpha} />
            <Field label="LoRA Dropout" type="number" value={loraDropout} onChange={setLoraDropout} />
          </div>
          <Field label="Target Modules (comma separated)" value={targetModules} onChange={setTargetModules} />
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input type="checkbox" checked={useQlora} onChange={(e) => setUseQlora(e.target.checked)} /> Use QLoRA 4-bit loading
          </label>
          <label className="flex items-center gap-2 text-xs text-slate-300">
            <input type="checkbox" checked={cleanupRemote} onChange={(e) => setCleanupRemote(e.target.checked)} /> Clean remote job workspace after download
          </label>
          <div className="text-[11px] text-slate-500 leading-relaxed">
            PDF files are converted locally into a temporary JSON dataset before upload. The temporary JSON file is deleted after the job exits.
          </div>
        </div>

        {latest?.status === "completed" && (
          <div className="mt-4 p-4 border-2 border-error bg-error/20 text-red-100 text-xs font-black animate-pulse">
            TRAINING FINISHED. Adapter saved to: <span className="font-mono">{latest.local_output_dir}</span>
            <br />
            AdCust has only released its SSH session. Your remote server may still be running and billing. Stop or terminate it on the provider website now.
          </div>
        )}

        <div className="mt-6 flex gap-2">
          <button disabled={!canStart} onClick={startTraining} className="btn-primary flex-1 h-11">
            <Play size={15} /> Start Training
          </button>
          <button disabled={!isTraining && !latest?.job_id} onClick={stopTraining} className="btn-danger w-32 h-11">
            <Square size={14} /> Abort SSH Job
          </button>
        </div>
      </section>

      <section className="min-w-0 space-y-4">
        <div className="grid grid-cols-2 2xl:grid-cols-4 gap-3">
          <Metric icon={<CheckCircle2 size={15} />} label="Job Status" value={latest?.status || "idle"} />
          <Metric icon={<Timer size={15} />} label="Elapsed" value={`${Math.round(latest?.elapsed_seconds || 0)}s`} />
          <Metric icon={<WalletCards size={15} />} label="Estimated Spend" value={`$${Number(latest?.estimated_cost_usd || 0).toFixed(4)}`} />
          <Metric icon={<FolderDown size={15} />} label="ETA" value={etaLabel || "waiting"} />
        </div>

        <div className="grid grid-cols-1 2xl:grid-cols-[260px_minmax(0,1fr)] items-start gap-4">
          <div className="border border-white/10 bg-surface-container-low p-4">
            <div className="text-xs font-bold mb-3">Workflow Gate</div>
            <div className="space-y-2">
              {workflow.map((step) => (
                <div key={step.label} className="flex items-center gap-2 text-[11px] text-slate-400">
                  <span className={`w-2 h-2 ${step.ok ? "bg-secondary" : "bg-error"}`} />
                  <span>{step.label}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="border border-white/10 bg-surface-container-low p-4 min-w-0">
            <div className="grid grid-cols-2 xl:grid-cols-4 gap-2 mb-3">
              <MiniStatus label="Stage" value={latest?.stage || "idle"} />
              <MiniStatus label="Train Progress" value={trainingProgress} />
              <MiniStatus label="Step" value={latest?.step && latest?.total_steps ? `${latest.step}/${latest.total_steps}` : "--"} />
              <MiniStatus
                label="Knowledge QA"
                value={typeof latest?.qa_pair_count === "number" || typeof latest?.qa_sample_count === "number"
                  ? String(latest.qa_pair_count ?? latest.qa_sample_count)
                  : "--"}
              />
            </div>
            <div className="h-2 bg-black/40 border border-white/10 overflow-hidden">
              <div className="h-full bg-primary transition-all" style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
            </div>
            <div className={`mt-3 text-sm flex items-start gap-2 ${latest?.error || error ? "text-red-200" : "text-slate-300"}`}>
              {latest?.error || error ? <AlertTriangle size={16} className="text-error mt-0.5" /> : <FolderDown size={16} className="text-secondary mt-0.5" />}
              <div className="min-w-0">
                <div className="font-bold">{error || latest?.message || "No training job running"}</div>
                {latest?.job_id && <div className="text-xs text-slate-500 mt-1 font-mono">Job: {latest.job_id}</div>}
                {latest?.local_output_dir && <div className="text-xs text-slate-500 mt-1 font-mono truncate">Local adapter: {latest.local_output_dir}</div>}
              </div>
              <button title="Refresh job status" disabled={!latest?.job_id} onClick={refreshJobStatus} className="icon-button ml-auto">
                <RefreshCw size={14} />
              </button>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)] items-start gap-4">
          <div className="border border-white/10 bg-surface-container-low p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-bold">Training Pipeline</div>
              <div className="text-[11px] font-mono text-primary">{progress.toFixed(1)}%</div>
            </div>
            <div className="space-y-1.5">
              {PIPELINE_STEPS.map((step) => {
                const state = getPipelineState(step, progress, latest?.status, cleanupRemote);
                return (
                  <div key={step.label} className="grid grid-cols-[18px_1fr_auto] items-center gap-2 min-h-7 text-[11px]">
                    <span className={`w-3 h-3 border ${state === "completed" ? "bg-secondary border-secondary" : state === "running" ? "bg-primary border-primary animate-pulse" : state === "failed" ? "bg-error border-error" : "border-white/20"}`} />
                    <span className={state === "pending" || state === "skipped" ? "text-slate-500" : "text-slate-200"}>{step.label}</span>
                    <span className={`uppercase text-[9px] font-bold ${state === "failed" ? "text-error" : state === "running" ? "text-primary" : state === "completed" ? "text-secondary" : "text-slate-600"}`}>{state}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="border border-white/10 bg-surface-container-low p-4 min-w-0">
            <div className="flex items-center justify-between mb-3">
              <div className="text-xs font-bold">Environment & Assets</div>
              <div className="text-[11px] text-slate-500">{resources.length} tracked items</div>
            </div>
            <div className="h-64 overflow-y-auto custom-scrollbar border border-white/5 bg-black/20">
              {resources.length === 0 ? (
                <div className="h-full flex items-center justify-center text-xs text-slate-600">Dependency and model file checks will appear here.</div>
              ) : resources.map((resource) => (
                <div key={resource.id} className="px-3 py-2 border-b border-white/5 last:border-b-0">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className={`w-2 h-2 shrink-0 ${resource.status === "ready" ? "bg-secondary" : resource.status === "missing" || resource.status === "mismatch" ? "bg-error" : resource.status.includes("download") ? "bg-primary animate-pulse" : "bg-slate-500"}`} />
                    <span className="text-[11px] font-medium truncate" title={resource.name}>{resource.name}</span>
                    <span className="ml-auto text-[9px] uppercase font-bold text-slate-500">{resource.status.replace(/_/g, " ")}</span>
                  </div>
                  <div className="mt-1 flex items-center gap-2 text-[9px] text-slate-600">
                    <span>{resource.kind.replace(/_/g, " ")}</span>
                    {typeof resource.size_bytes === "number" && <span>{formatBytes(resource.size_bytes)}</span>}
                    {typeof resource.download_progress === "number" && <span>{resource.download_progress.toFixed(1)}%</span>}
                  </div>
                  {typeof resource.download_progress === "number" && (
                    <div className="h-1 bg-black/50 mt-1 overflow-hidden">
                      <div className="h-full bg-primary" style={{ width: `${Math.max(0, Math.min(100, resource.download_progress))}%` }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="border border-white/10 bg-surface-container-low p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-bold">Loss Curve</div>
              <div className="text-[11px] text-slate-500">
                {lossPoints.length
                  ? `${lossPoints.length} points · latest ${lossPoints[lossPoints.length - 1].loss.toFixed(4)}`
                  : "waiting for training metrics"}
              </div>
          </div>
          <LossChart points={lossPoints} />
        </div>

        <div className="border border-white/10 bg-surface-container-low p-4">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-bold">Training Logs</div>
            <button type="button" onClick={copyLogs} disabled={logs.length === 0} className="btn-tool h-8 px-3 text-[11px]">
              <ClipboardCopy size={14} /> Copy Logs
            </button>
          </div>
          <div className="h-72 bg-black border border-white/10 overflow-y-auto custom-scrollbar p-4 font-mono text-[11px] select-text whitespace-pre-wrap">
            {logs.length === 0 ? (
              <div className="text-slate-600">Remote training logs will stream here after the job starts.</div>
            ) : logs.map((line, index) => (
              <div key={index} className={line.includes("failed") || line.includes("aborted") ? "text-red-300" : "text-slate-400"}>{line}</div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}

function MiniStatus({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-black/20 p-2 min-w-0">
      <div className="text-[10px] text-slate-500 uppercase">{label}</div>
      <div className="text-xs font-bold truncate" title={value}>{value}</div>
    </div>
  );
}

function LossChart({ points }: { points: { step: number; loss: number }[] }) {
  if (points.length === 0) {
    return <div className="h-32 border border-dashed border-white/10 flex items-center justify-center text-xs text-slate-600">Loss metrics will appear after the trainer starts logging.</div>;
  }
  const width = 520;
  const height = 120;
  const losses = points.map((p) => p.loss);
  const min = Math.min(...losses);
  const max = Math.max(...losses);
  const span = Math.max(0.000001, max - min);
  const polyline = points.map((p, index) => {
    const x = (index / Math.max(1, points.length - 1)) * width;
    const y = height - ((p.loss - min) / span) * (height - 12) - 6;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-32 border border-white/10 bg-black/30">
      {points.length > 1 && <polyline points={polyline} fill="none" stroke="rgb(66, 153, 225)" strokeWidth="2" />}
      {points.map((point, index) => {
        const x = points.length === 1 ? width / 2 : (index / (points.length - 1)) * width;
        const y = height - ((point.loss - min) / span) * (height - 12) - 6;
        return <circle key={`${point.step}-${index}`} cx={x} cy={y} r="3.5" fill="rgb(96, 165, 250)" />;
      })}
    </svg>
  );
}

function formatDuration(seconds?: number) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "";
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  if (minutes <= 0) return `${secs}s`;
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  return hours > 0 ? `${hours}h ${mins}m` : `${mins}m ${secs}s`;
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function getPipelineState(
  step: { start: number; complete: number; optional?: boolean },
  progress: number,
  jobStatus?: string,
  cleanupRemote = false,
) {
  if (step.optional && !cleanupRemote && progress >= step.start) return "skipped";
  if (jobStatus === "failed" && progress >= step.start && progress < step.complete) return "failed";
  if (progress >= step.complete) return "completed";
  if (progress >= step.start) return "running";
  return "pending";
}

function PanelTitle({ icon, text }: { icon: ReactNode; text: string }) {
  return <div className="flex items-center gap-2 text-xs font-bold text-slate-300 pt-2">{icon}{text}</div>;
}

function Field({ label, value, onChange, type = "text" }: { label: string; value: string; type?: string; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="text-[10px] text-slate-500 uppercase">{label}</span>
      <input type={type} value={value} onChange={(e) => onChange(e.target.value)} className="mt-1 w-full h-10 bg-black/40 border border-white/10 px-3 text-xs outline-none focus:border-primary" />
    </label>
  );
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-surface-container-low p-3 min-w-0">
      <div className="flex items-center gap-2 text-[10px] text-slate-500 mb-1">{icon}{label}</div>
      <div className="text-sm font-bold truncate" title={value}>{value}</div>
    </div>
  );
}
