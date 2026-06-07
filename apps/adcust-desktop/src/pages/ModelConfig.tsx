import { Cpu, FolderOpen } from "lucide-react";

export default function ModelConfig() {
  return (
    <div className="p-8 max-w-2xl">
      <div className="space-y-6">
        <h2 className="text-xl font-black flex items-center gap-3"><Cpu /> MODEL_ASSET_MANAGER</h2>
        
        <div className="p-6 bg-surface-dark border border-cyber-green/10 space-y-4">
          <p className="text-xs opacity-60">Select a local HuggingFace model folder.</p>
          <button className="file-picker-button py-8">
            <FolderOpen className="group-hover:text-neon-green" />
            <span className="font-bold">SELECT_MODEL_PATH</span>
          </button>
        </div>
      </div>
    </div>
  );
}
