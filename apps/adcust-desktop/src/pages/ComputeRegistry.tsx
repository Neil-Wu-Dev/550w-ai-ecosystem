import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { Activity, AlertTriangle, CheckCircle2, Pencil, PlugZap, Plus, RefreshCw, Server, Trash2, X } from "lucide-react";

type PageProps = {
  apiBase: string;
  backendReady: boolean;
};

type Provider = {
  id: string;
  name: string;
  provider_type: string;
  endpoint_summary: string;
  connection_info: Record<string, any>;
  remote_workspace_path: string;
  hourly_rate_usd: number;
  is_active: boolean;
  telemetry_data: any;
  last_heartbeat?: string;
};

const emptyForm = {
  id: "",
  name: "",
  host: "",
  port: "",
  username: "",
  password: "",
  key_path: "",
  remote_workspace_path: "",
  hourly_rate_usd: "",
};

const ACTIVE_COMPUTE_KEY = "adcust.activeComputeNodeId";

export default function ComputeRegistry({ apiBase, backendReady }: PageProps) {
  const [providers, setProviders] = useState<Provider[]>([]);
  const [showModal, setShowModal] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [message, setMessage] = useState("Waiting for backend configuration");
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [activeComputeId, setActiveComputeId] = useState(() => localStorage.getItem(ACTIVE_COMPUTE_KEY) || "");
  const [formData, setFormData] = useState(emptyForm);

  const computeApi = useMemo(() => (apiBase ? `${apiBase}/api/v1/compute` : ""), [apiBase]);

  const explainError = async (res: Response) => {
    const data = await res.json().catch(() => ({}));
    return data.message || data.detail || data.error || `${res.status} ${res.statusText}`;
  };

  const fetchProviders = async () => {
    if (!computeApi) {
      setProviders([]);
      setMessage("Configure the backend base URL before loading compute nodes");
      return;
    }
    try {
      const res = await fetch(`${computeApi}/`);
      if (!res.ok) throw new Error(await explainError(res));
      const items = await res.json();
      setProviders(items);
      if (!activeComputeId && items.length === 1) {
        setActiveComputeId(items[0].id);
        localStorage.setItem(ACTIVE_COMPUTE_KEY, items[0].id);
      }
      setMessage("Compute inventory synchronized");
    } catch (err: any) {
      setError(err.message);
      setMessage("Compute inventory request failed");
    }
  };

  const openCreateModal = () => {
    setEditingId(null);
    setFormData(emptyForm);
    setShowModal(true);
  };

  const openEditModal = (provider: Provider) => {
    const info = provider.connection_info || {};
    setEditingId(provider.id);
    setFormData({
      id: provider.id,
      name: provider.name,
      host: String(info.host || ""),
      port: String(info.port || ""),
      username: String(info.username || ""),
      password: String(info.password || info.passphrase || ""),
      key_path: String(info.key_path || ""),
      remote_workspace_path: String(info.remote_workspace_path || provider.remote_workspace_path || ""),
      hourly_rate_usd: String(info.hourly_rate_usd ?? provider.hourly_rate_usd ?? ""),
    });
    setShowModal(true);
  };

  const saveProvider = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!computeApi) return setError("Backend base URL is not configured");
    setError(null);
    const payload = {
      id: formData.id.trim(),
      name: formData.name.trim(),
      provider_type: "SSH",
      connection_info: {
        host: formData.host.trim(),
        port: Number(formData.port),
        username: formData.username.trim(),
        password: formData.password || undefined,
        key_path: formData.key_path || undefined,
        remote_workspace_path: formData.remote_workspace_path.trim(),
        hourly_rate_usd: Number(formData.hourly_rate_usd),
      },
    };
    try {
      const res = await fetch(editingId ? `${computeApi}/${editingId}` : `${computeApi}/`, {
        method: editingId ? "PUT" : "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await explainError(res));
      setShowModal(false);
      setEditingId(null);
      setFormData(emptyForm);
      setMessage(editingId ? "Node settings saved. Click Connect to test SSH again." : "Node registered. Start the server on the provider website, then click Connect.");
      await fetchProviders();
    } catch (err: any) {
      setError(err.message);
    }
  };

  const operateNode = async (id: string, action: "verify" | "sync" | "delete") => {
    if (!computeApi) return setError("Backend base URL is not configured");
    setBusyId(id);
    setError(null);
    setSuccess(null);
    try {
      const url = action === "delete" ? `${computeApi}/${id}` : `${computeApi}/${id}/${action}`;
      const res = await fetch(url, { method: action === "delete" ? "DELETE" : "POST" });
      if (!res.ok) throw new Error(await explainError(res));
      const data = await res.json().catch(() => ({}));
      if (action === "verify") {
        setSuccess(data.status === "connected" ? "Connection successful. SSH is reachable." : null);
        setMessage(`SSH connect attempt: ${data.status}`);
      } else if (action === "sync") {
        setSuccess(data.is_active ? "Connection check successful. Server is reachable." : null);
        setMessage(data.is_active ? "Connection status checked: connected" : "Connection status checked: disconnected");
      } else {
        if (activeComputeId === id) {
          setActiveComputeId("");
          localStorage.removeItem(ACTIVE_COMPUTE_KEY);
        }
        setMessage("Node removed");
      }
      await fetchProviders();
    } catch (err: any) {
      setError(err.message);
      await fetchProviders().catch(() => null);
    } finally {
      setBusyId(null);
    }
  };

  const useNode = (id: string) => {
    setActiveComputeId(id);
    localStorage.setItem(ACTIVE_COMPUTE_KEY, id);
    setSuccess("Compute node selected for training.");
    setError(null);
  };

  const checkAllProviders = async () => {
    if (!computeApi) return;
    await Promise.all(providers.map((provider) => fetch(`${computeApi}/${provider.id}/sync`, { method: "POST" }).catch(() => null)));
    if (providers.length > 0) await fetchProviders().catch(() => null);
  };

  useEffect(() => {
    fetchProviders();
    const timer = window.setInterval(fetchProviders, 8000);
    return () => window.clearInterval(timer);
  }, [computeApi]);

  useEffect(() => {
    const timer = window.setInterval(() => checkAllProviders(), 30000);
    return () => window.clearInterval(timer);
  }, [computeApi, providers]);

  return (
    <div className="h-full flex flex-col gap-5">
      <section className="flex items-center justify-between border-b border-white/10 pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight">Remote Compute Nodes</h1>
          <p className="text-xs text-slate-500 mt-1">Register Linux GPU hosts. AdCust can connect over SSH and check reachability; it cannot start, stop, destroy, or stop billing for servers.</p>
        </div>
        <button disabled={!backendReady} onClick={openCreateModal} className="btn-primary">
          <Plus size={15} /> Add Node
        </button>
      </section>

      <StatusBanner error={error} success={success} message={error || success || message} />

      <div className="border-2 border-error/60 bg-error/15 px-4 py-3 text-xs text-red-100 flex items-start gap-3">
        <AlertTriangle size={18} className="text-error shrink-0 mt-0.5" />
        <div className="leading-5">
          <div className="font-black uppercase tracking-wide">Manual server lifecycle required</div>
          <div>You must start the server on the provider website before connecting. Disconnecting AdCust only releases the SSH session. Use Check Status to detect whether a server is still reachable.</div>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {providers.map((p) => {
          const env = p.telemetry_data?.env_probe || {};
          const session = p.telemetry_data?.session || {};
          const gpu = env.gpus?.[0];
          return (
            <article key={p.id} className={`border bg-surface-container-low p-4 ${activeComputeId === p.id ? "border-primary/70" : "border-white/10"}`}>
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-start gap-3 min-w-0">
                  <div className="w-10 h-10 bg-surface-container-high flex items-center justify-center border border-white/10">
                    <Server size={18} className={p.is_active ? "text-secondary" : "text-slate-500"} />
                  </div>
                  <div className="min-w-0">
                    <h2 className="font-bold truncate">{p.name}</h2>
                    <p className="text-xs text-slate-500 font-mono mt-1 truncate">{p.endpoint_summary}</p>
                    <p className="text-xs text-slate-500 font-mono mt-1 truncate">{p.remote_workspace_path}</p>
                  </div>
                </div>
                <span className={`text-[10px] px-2 py-1 font-bold ${p.is_active ? "bg-secondary/10 text-secondary" : "bg-white/5 text-slate-500"}`}>
                  {p.is_active ? "CONNECTED" : "DISCONNECTED"}
                </span>
              </div>

              {activeComputeId === p.id && (
                <div className="mt-3 border border-primary/40 bg-primary/10 text-primary px-3 py-2 text-[11px] font-bold">
                  IN USE FOR TRAINING
                </div>
              )}

              <div className="grid grid-cols-4 gap-px bg-white/10 mt-4 border border-white/10">
                <Metric label="Connection" value={p.is_active ? "Connected" : "Disconnected"} />
                <Metric label="Free VRAM" value={gpu ? `${gpu.memory_free_mb} MB` : "--"} />
                <Metric label="PyTorch" value={env.has_pytorch ? env.torch_version || "Available" : "Not synced"} />
                <Metric label="Rate" value={`$${Number(p.hourly_rate_usd || 0).toFixed(4)}/h`} />
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 text-[11px] text-slate-500">
                <div>Disk free: {env.disk_free_gb ? `${env.disk_free_gb} GB` : "--"}</div>
                <div>GPU temp: {gpu?.temperature_c !== undefined ? `${gpu.temperature_c} C` : "--"}</div>
                <div>GPU util: {gpu?.utilization_percent !== undefined ? `${gpu.utilization_percent}%` : "--"}</div>
                <div>Last heartbeat: {p.last_heartbeat ? new Date(p.last_heartbeat).toLocaleString() : "Never"}</div>
                {session.note && <div className="col-span-2 text-warning">{session.note}</div>}
                <div className="col-span-2 text-error font-bold">
                  Billing warning: AdCust cannot stop this server. Stop or terminate it on the provider website.
                </div>
              </div>

              <div className="flex justify-end gap-2 mt-4 flex-wrap">
                <ActionButton label="Connect" title="Connect: try SSH to this running server" disabled={busyId === p.id} onClick={() => operateNode(p.id, "verify")} icon={<PlugZap size={14} />} />
                <ActionButton label="Check Status" title="Check Status: test whether SSH is still reachable" disabled={busyId === p.id} onClick={() => operateNode(p.id, "sync")} icon={<RefreshCw size={14} className={busyId === p.id ? "animate-spin" : ""} />} />
                <ActionButton label={activeComputeId === p.id ? "In Use" : "Use"} title="Use this compute node for training" disabled={busyId === p.id || activeComputeId === p.id} onClick={() => useNode(p.id)} icon={<CheckCircle2 size={14} />} />
                <ActionButton label="Edit" title="Edit saved SSH connection settings" disabled={busyId === p.id} onClick={() => openEditModal(p)} icon={<Pencil size={14} />} />
                <ActionButton label="Remove" title="Remove node" disabled={busyId === p.id} onClick={() => operateNode(p.id, "delete")} icon={<Trash2 size={14} />} danger />
              </div>
            </article>
          );
        })}
      </div>

      {providers.length === 0 && (
        <div className="flex-1 border border-dashed border-white/10 flex items-center justify-center text-slate-500 text-sm">
          No compute nodes registered.
        </div>
      )}

      {showModal && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-6">
          <form onSubmit={saveProvider} className="w-[600px] bg-[#161618] border border-white/10">
            <div className="h-12 px-4 border-b border-white/10 flex items-center justify-between">
              <span className="font-bold text-sm">{editingId ? "Edit SSH Compute Node" : "Add SSH Compute Node"}</span>
              <button type="button" onClick={() => setShowModal(false)} className="icon-button"><X size={16} /></button>
            </div>
            <div className="p-5 grid grid-cols-2 gap-4">
              <Field label="Node ID" required disabled={Boolean(editingId)} value={formData.id} onChange={(id) => setFormData({ ...formData, id })} />
              <Field label="Display Name" required value={formData.name} onChange={(name) => setFormData({ ...formData, name })} />
              <Field label="Host / IP" required value={formData.host} onChange={(host) => setFormData({ ...formData, host })} />
              <Field label="SSH Port" required type="number" value={formData.port} onChange={(port) => setFormData({ ...formData, port })} />
              <Field label="Username" required value={formData.username} onChange={(username) => setFormData({ ...formData, username })} />
              <Field label="Hourly Cost USD" required type="number" value={formData.hourly_rate_usd} onChange={(hourly_rate_usd) => setFormData({ ...formData, hourly_rate_usd })} />
              <Field label="SSH Key Path" value={formData.key_path} onChange={(key_path) => setFormData({ ...formData, key_path })} />
              <Field label="Password / Key Passphrase" value={formData.password} onChange={(password) => setFormData({ ...formData, password })} />
              <div className="col-span-2">
                <Field label="Remote Workspace Path" required value={formData.remote_workspace_path} onChange={(remote_workspace_path) => setFormData({ ...formData, remote_workspace_path })} />
              </div>
            </div>
            <div className="p-4 border-t border-white/10 flex justify-end gap-2">
              <button type="button" onClick={() => setShowModal(false)} className="btn-tool">Cancel</button>
              <button type="submit" className="btn-primary"><CheckCircle2 size={14} /> {editingId ? "Save Changes" : "Save Node"}</button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

function StatusBanner({ error, success, message }: { error: string | null; success: string | null; message: string }) {
  const mode = error ? "error" : success ? "success" : "neutral";
  return (
    <div className={`border px-4 py-3 text-xs flex items-center gap-2 ${mode === "error" ? "border-error/50 bg-error/15 text-red-100" : mode === "success" ? "border-secondary/50 bg-secondary/10 text-secondary" : "border-white/10 bg-surface-container-low text-slate-300"}`}>
      {error ? <AlertTriangle size={16} className="text-error" /> : success ? <CheckCircle2 size={16} className="text-secondary" /> : <Activity size={16} className="text-slate-400" />}
      <span className="font-mono">{message}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-[#151517] p-3 min-w-0">
      <div className="text-[10px] text-slate-500 mb-1">{label}</div>
      <div className="text-xs text-slate-200 truncate" title={value}>{value}</div>
    </div>
  );
}

function Field({ label, value, onChange, type = "text", required = false, disabled = false }: { label: string; value: string; type?: string; required?: boolean; disabled?: boolean; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="text-[10px] text-slate-500 uppercase">{label}</span>
      <input disabled={disabled} required={required} type={type} value={value} onChange={(e) => onChange(e.target.value)} className="mt-1 w-full h-9 bg-black/40 border border-white/10 px-3 text-xs outline-none focus:border-primary disabled:opacity-60 disabled:cursor-not-allowed" />
    </label>
  );
}

function ActionButton({ icon, label, title, onClick, disabled, danger = false }: { icon: ReactNode; label: string; title: string; onClick: () => void; disabled?: boolean; danger?: boolean }) {
  return (
    <button title={title} disabled={disabled} onClick={onClick} className={`btn-tool h-8 px-3 text-[11px] ${danger ? "hover:!border-error hover:!bg-error/20" : ""}`}>
      {icon}
      {label}
    </button>
  );
}
