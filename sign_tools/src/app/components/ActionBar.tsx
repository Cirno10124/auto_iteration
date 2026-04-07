import { CheckCircle2, SkipForward, ChevronRight, AlertCircle } from 'lucide-react';

interface ActionBarProps {
  onAccept: () => void;
  onSubmitNext: () => void;
  onSkip: () => void;
  onMarkReview: () => void;
  currentIndex: number;
  total: number;
  isLastItem: boolean;
}

export function ActionBar({
  onAccept,
  onSubmitNext,
  onSkip,
  onMarkReview,
  currentIndex,
  total,
  isLastItem,
}: ActionBarProps) {
  return (
    <footer className="h-16 bg-[#13161f] border-t border-[#252836] flex items-center px-6 gap-3 shrink-0">
      {/* Progress pill */}
      <div className="text-xs text-[#4a5068] mr-2">
        <span className="text-[#94a3b8]">{currentIndex + 1}</span>
        <span className="mx-1">/</span>
        <span>{total}</span>
      </div>

      {/* Mark review (secondary) */}
      <button
        onClick={onMarkReview}
        title="标记待复核"
        className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#252836] text-[#a78bfa] hover:bg-[#a78bfa]/10 hover:border-[#a78bfa]/40 transition-all text-sm"
      >
        <AlertCircle size={15} />
        <span>待复核</span>
      </button>

      {/* Skip */}
      <button
        onClick={onSkip}
        title="跳过 (S)"
        className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#252836] text-[#f59e0b] hover:bg-[#f59e0b]/10 hover:border-[#f59e0b]/40 transition-all text-sm"
      >
        <SkipForward size={15} />
        <span>跳过</span>
        <kbd className="text-[10px] bg-[#252836] px-1.5 py-0.5 rounded text-[#4a5068]">S</kbd>
      </button>

      <div className="flex-1" />

      {/* Accept (without advancing) */}
      <button
        onClick={onAccept}
        title="接受当前标签"
        className="flex items-center gap-2 px-4 py-2 rounded-lg border border-[#22c55e]/40 text-[#22c55e] hover:bg-[#22c55e]/10 transition-all text-sm"
      >
        <CheckCircle2 size={15} />
        <span>接受</span>
      </button>

      {/* Submit & Next (primary CTA) */}
      <button
        onClick={onSubmitNext}
        title="保存并下一条 (Enter)"
        className="flex items-center gap-2 px-6 py-2 rounded-lg bg-[#4f7af8] hover:bg-[#6b8ffa] text-white transition-all text-sm shadow-lg shadow-[#4f7af8]/20 active:scale-95"
      >
        <span>{isLastItem ? '保存完成' : '保存并下一条'}</span>
        {!isLastItem && <ChevronRight size={15} />}
        <kbd className="text-[10px] bg-white/15 px-1.5 py-0.5 rounded">Enter</kbd>
      </button>
    </footer>
  );
}
