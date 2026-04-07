import { X, FileJson, FileSpreadsheet, Upload, Download, CheckCircle2 } from 'lucide-react';
import { useState } from 'react';
import { TaskItem } from '../types';

interface ImportExportModalProps {
  mode: 'import' | 'export';
  tasks: TaskItem[];
  onClose: () => void;
  onImport: (tasks: TaskItem[]) => void;
}

export function ImportExportModal({ mode, tasks, onClose, onImport }: ImportExportModalProps) {
  const [exportFormat, setExportFormat] = useState<'json' | 'csv'>('json');
  const [done, setDone] = useState(false);

  const handleExport = () => {
    const annotated = tasks.filter((t) => t.status !== 'pending');
    if (exportFormat === 'json') {
      const data = annotated.map((t) => ({
        audio_id: t.id,
        audio_filename: t.audioFilename,
        final_text: t.finalText,
        label_level: t.labelGrade ?? null,
        skipped: t.status === 'skipped',
        status: t.status,
        issue_tag: t.issueTag ?? null,
        confidence: t.confidence,
        batch_id: t.batchId,
      }));
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotations_${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
    } else {
      const headers = 'audio_id,audio_filename,final_text,label_level,skipped,status,issue_tag,confidence,batch_id';
      const rows = annotated.map((t) =>
        [
          t.id,
          t.audioFilename,
          `"${t.finalText.replace(/"/g, '""')}"`,
          t.labelGrade ?? '',
          t.status === 'skipped',
          t.status,
          t.issueTag ?? '',
          t.confidence,
          t.batchId,
        ].join(',')
      );
      const csv = [headers, ...rows].join('\n');
      const blob = new Blob([csv], { type: 'text/csv' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `annotations_${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
    }
    setDone(true);
    setTimeout(() => { setDone(false); onClose(); }, 1500);
  };

  const handleFileImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => {
      try {
        const data = JSON.parse(ev.target?.result as string);
        // Map imported data to TaskItem format (simplified)
        if (Array.isArray(data)) {
          alert(`已读取 ${data.length} 条任务（演示模式，实际不修改数据）`);
        }
      } catch {
        alert('文件格式不正确，请使用 JSON 格式');
      }
    };
    reader.readAsText(file);
    onClose();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-[#13161f] border border-[#252836] rounded-2xl w-[420px] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#252836]">
          <div className="flex items-center gap-2">
            {mode === 'import' ? <Upload size={16} className="text-[#4f7af8]" /> : <Download size={16} className="text-[#22c55e]" />}
            <h2 className="text-[#e2e8f0] text-base">{mode === 'import' ? '导入任务' : '导出标注结果'}</h2>
          </div>
          <button onClick={onClose} className="w-8 h-8 flex items-center justify-center text-[#4a5068] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all">
            <X size={16} />
          </button>
        </div>

        <div className="p-6">
          {mode === 'import' ? (
            <div className="flex flex-col gap-4">
              <p className="text-sm text-[#94a3b8]">支持导入 JSON 格式的任务列表，字段需包含：</p>
              <div className="bg-[#0c0e14] rounded-xl p-4 text-xs text-[#4a5068] font-mono leading-relaxed">
                <div className="text-[#4f7af8]">{'{'}</div>
                <div className="pl-4">
                  <div><span className="text-[#a78bfa]">"audio_path"</span><span className="text-[#94a3b8]">: "./audio/file.wav"</span><span className="text-[#4a5068]">,</span></div>
                  <div><span className="text-[#a78bfa]">"pseudo_label"</span><span className="text-[#94a3b8]">: "模型输出文本"</span><span className="text-[#4a5068]">,</span></div>
                  <div><span className="text-[#a78bfa]">"confidence"</span><span className="text-[#94a3b8]">: 0.82</span><span className="text-[#4a5068]">,</span></div>
                  <div><span className="text-[#a78bfa]">"batch_id"</span><span className="text-[#94a3b8]">: "batch_03"</span></div>
                </div>
                <div className="text-[#4f7af8]">{'}'}</div>
              </div>
              <label className="flex flex-col items-center gap-3 py-8 border-2 border-dashed border-[#252836] rounded-xl cursor-pointer hover:border-[#4f7af8]/50 hover:bg-[#4f7af8]/5 transition-all">
                <Upload size={24} className="text-[#4a5068]" />
                <span className="text-sm text-[#4a5068]">点击选择 JSON 文件</span>
                <input type="file" accept=".json" className="hidden" onChange={handleFileImport} />
              </label>
            </div>
          ) : (
            <div className="flex flex-col gap-4">
              <div>
                <div className="text-xs text-[#4a5068] mb-2">导出格式</div>
                <div className="flex gap-2">
                  {(['json', 'csv'] as const).map((fmt) => (
                    <button
                      key={fmt}
                      onClick={() => setExportFormat(fmt)}
                      className={`flex items-center gap-2 px-4 py-3 rounded-xl border flex-1 transition-all ${
                        exportFormat === fmt
                          ? 'bg-[#4f7af8]/10 border-[#4f7af8]/50 text-[#4f7af8]'
                          : 'border-[#252836] text-[#4a5068] hover:border-[#3a3d4a]'
                      }`}
                    >
                      {fmt === 'json' ? <FileJson size={16} /> : <FileSpreadsheet size={16} />}
                      <div className="text-left">
                        <div className="text-sm">{fmt.toUpperCase()}</div>
                        <div className="text-[10px] opacity-70">{fmt === 'json' ? '结构化数据' : '表格格式'}</div>
                      </div>
                    </button>
                  ))}
                </div>
              </div>
              <div className="bg-[#0c0e14] rounded-xl p-3 text-xs text-[#4a5068]">
                将导出 <span className="text-[#e2e8f0]">{tasks.filter((t) => t.status !== 'pending').length}</span> 条已标注结果
                （共 {tasks.length} 条，{tasks.filter((t) => t.status === 'pending').length} 条未完成）
              </div>
              <button
                onClick={handleExport}
                className="flex items-center justify-center gap-2 py-3 bg-[#22c55e] hover:bg-[#16a34a] text-white rounded-xl transition-all text-sm font-medium"
              >
                {done ? (
                  <>
                    <CheckCircle2 size={16} />
                    导出成功！
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    导出 {exportFormat.toUpperCase()} 文件
                  </>
                )}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
