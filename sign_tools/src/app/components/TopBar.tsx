import { useState } from 'react';
import {
  Keyboard,
  Download,
  Upload,
  ChevronDown,
  BarChart2,
  Settings,
  Mic2,
} from 'lucide-react';
import { FilterStatus, SessionStats, SortMode } from '../types';

interface TopBarProps {
  stats: SessionStats;
  filterStatus: FilterStatus;
  sortMode: SortMode;
  onFilterChange: (f: FilterStatus) => void;
  onSortChange: (s: SortMode) => void;
  onShortcutsOpen: () => void;
  onImport: () => void;
  onExport: () => void;
}

const FILTER_TABS: { label: string; value: FilterStatus; color: string }[] = [
  { label: '全部', value: 'all', color: '' },
  { label: '待标注', value: 'pending', color: 'text-[#94a3b8]' },
  { label: '已接受', value: 'accepted', color: 'text-[#22c55e]' },
  { label: '已跳过', value: 'skipped', color: 'text-[#f59e0b]' },
  { label: '待复核', value: 'review', color: 'text-[#a78bfa]' },
];

const SORT_OPTIONS: { label: string; value: SortMode }[] = [
  { label: '置信度↑（最不确定优先）', value: 'confidence_asc' },
  { label: '中等置信度优先', value: 'confidence_mid' },
  { label: '顺序排列', value: 'sequential' },
];

export function TopBar({
  stats,
  filterStatus,
  sortMode,
  onFilterChange,
  onSortChange,
  onShortcutsOpen,
  onImport,
  onExport,
}: TopBarProps) {
  const [sortOpen, setSortOpen] = useState(false);

  const progressPct = stats.total > 0 ? Math.round(((stats.accepted + stats.skipped) / stats.total) * 100) : 0;

  return (
    <header className="h-14 bg-[#13161f] border-b border-[#252836] flex items-center px-4 gap-4 shrink-0 z-20">
      {/* Logo */}
      <div className="flex items-center gap-2 min-w-[180px]">
        <div className="w-7 h-7 rounded-lg bg-[#4f7af8] flex items-center justify-center">
          <Mic2 size={14} className="text-white" />
        </div>
        <div>
          <div className="text-[#e2e8f0] text-sm leading-none">ASR 标注工具</div>
          <div className="text-[#4a5068] text-xs leading-none mt-0.5">batch_03</div>
        </div>
      </div>

      {/* Filter tabs */}
      <div className="flex items-center gap-1 bg-[#0c0e14] rounded-lg p-1">
        {FILTER_TABS.map((tab) => (
          <button
            key={tab.value}
            onClick={() => onFilterChange(tab.value)}
            className={`px-3 py-1 rounded-md text-xs transition-all ${
              filterStatus === tab.value
                ? 'bg-[#252836] text-[#e2e8f0] shadow'
                : 'text-[#4a5068] hover:text-[#94a3b8]'
            }`}
          >
            {tab.label}
            {tab.value === 'pending' && (
              <span className="ml-1.5 bg-[#4f7af8]/20 text-[#4f7af8] text-[10px] px-1.5 py-0.5 rounded-full">
                {stats.pending}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Sort dropdown */}
      <div className="relative">
        <button
          onClick={() => setSortOpen(!sortOpen)}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-[#0c0e14] hover:bg-[#252836] rounded-lg text-xs text-[#94a3b8] transition-all border border-[#252836]"
        >
          <BarChart2 size={12} />
          {SORT_OPTIONS.find((o) => o.value === sortMode)?.label ?? '排序'}
          <ChevronDown size={12} />
        </button>
        {sortOpen && (
          <div className="absolute top-9 left-0 bg-[#1b1e2b] border border-[#252836] rounded-lg shadow-xl z-50 py-1 min-w-[220px]">
            {SORT_OPTIONS.map((opt) => (
              <button
                key={opt.value}
                onClick={() => { onSortChange(opt.value); setSortOpen(false); }}
                className={`w-full text-left px-3 py-2 text-xs hover:bg-[#252836] transition-all ${
                  sortMode === opt.value ? 'text-[#4f7af8]' : 'text-[#94a3b8]'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Progress */}
      <div className="flex-1 flex items-center gap-3">
        <div className="text-xs text-[#4a5068]">
          <span className="text-[#e2e8f0]">{stats.accepted + stats.skipped}</span>
          <span className="mx-1">/</span>
          <span>{stats.total}</span>
          <span className="ml-2 text-[#22c55e]">✓ {stats.accepted}</span>
          <span className="ml-2 text-[#f59e0b]">→ {stats.skipped}</span>
        </div>
        <div className="flex-1 max-w-[200px] h-1.5 bg-[#252836] rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[#4f7af8] to-[#22c55e] rounded-full transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
        <div className="text-xs text-[#4a5068]">{progressPct}%</div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={onImport}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all"
        >
          <Upload size={12} />
          导入
        </button>
        <button
          onClick={onExport}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all"
        >
          <Download size={12} />
          导出
        </button>
        <button
          onClick={onShortcutsOpen}
          title="快捷键 (?)"
          className="w-8 h-8 flex items-center justify-center text-[#4a5068] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all"
        >
          <Keyboard size={16} />
        </button>
        <button className="w-8 h-8 flex items-center justify-center text-[#4a5068] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all">
          <Settings size={16} />
        </button>
      </div>
    </header>
  );
}
