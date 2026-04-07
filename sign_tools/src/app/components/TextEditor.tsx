import { useState, useRef, useEffect } from 'react';
import { RotateCcw, Copy, Check, Eye, EyeOff, Sparkles } from 'lucide-react';

interface TextEditorProps {
  pseudoLabel: string; // original model output
  value: string;
  onChange: (v: string) => void;
  onReset: () => void;
}

function diffWords(original: string, edited: string) {
  // Simple character-level diff for display
  if (original === edited) return null;

  const origWords = original.split('');
  const editWords = edited.split('');

  // Find common prefix
  let prefixLen = 0;
  while (
    prefixLen < origWords.length &&
    prefixLen < editWords.length &&
    origWords[prefixLen] === editWords[prefixLen]
  ) {
    prefixLen++;
  }

  // Find common suffix
  let origSuffix = origWords.length - 1;
  let editSuffix = editWords.length - 1;
  while (
    origSuffix >= prefixLen &&
    editSuffix >= prefixLen &&
    origWords[origSuffix] === editWords[editSuffix]
  ) {
    origSuffix--;
    editSuffix--;
  }

  return {
    prefix: original.slice(0, prefixLen),
    removed: original.slice(prefixLen, origSuffix + 1),
    added: edited.slice(prefixLen, editSuffix + 1),
    suffix: original.slice(origSuffix + 1),
  };
}

export function TextEditor({ pseudoLabel, value, onChange, onReset }: TextEditorProps) {
  const [showDiff, setShowDiff] = useState(false);
  const [copied, setCopied] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const isModified = value !== pseudoLabel;

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  const handleCopy = () => {
    navigator.clipboard.writeText(value);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const diff = isModified ? diffWords(pseudoLabel, value) : null;

  return (
    <div className="flex flex-col gap-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles size={14} className="text-[#4f7af8]" />
          <span className="text-sm text-[#94a3b8]">标注文本</span>
          {isModified && (
            <span className="text-[10px] bg-[#f59e0b]/15 text-[#f59e0b] px-2 py-0.5 rounded-full border border-[#f59e0b]/30">
              已修改
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {isModified && (
            <button
              onClick={() => setShowDiff(!showDiff)}
              className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs transition-all ${
                showDiff
                  ? 'bg-[#4f7af8]/15 text-[#4f7af8]'
                  : 'text-[#4a5068] hover:text-[#94a3b8] hover:bg-[#252836]'
              }`}
            >
              {showDiff ? <EyeOff size={12} /> : <Eye size={12} />}
              对比
            </button>
          )}
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-[#4a5068] hover:text-[#94a3b8] hover:bg-[#252836] transition-all"
          >
            {copied ? <Check size={12} className="text-[#22c55e]" /> : <Copy size={12} />}
            {copied ? '已复制' : '复制'}
          </button>
          {isModified && (
            <button
              onClick={onReset}
              title="恢复模型输出"
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs text-[#4a5068] hover:text-[#f59e0b] hover:bg-[#f59e0b]/10 transition-all"
            >
              <RotateCcw size={12} />
              恢复
            </button>
          )}
        </div>
      </div>

      {/* Original model output (diff view) */}
      {showDiff && isModified && (
        <div className="bg-[#0c0e14] border border-[#252836] rounded-xl p-4">
          <div className="text-xs text-[#4a5068] mb-2">模型原始输出</div>
          {diff ? (
            <div className="text-sm leading-relaxed text-[#94a3b8]">
              <span>{diff.prefix}</span>
              {diff.removed && (
                <span className="bg-[#ef4444]/20 text-[#ef4444] line-through px-0.5 rounded">
                  {diff.removed}
                </span>
              )}
              {diff.added && (
                <span className="bg-[#22c55e]/20 text-[#22c55e] px-0.5 rounded">
                  {diff.added}
                </span>
              )}
              <span>{diff.suffix}</span>
            </div>
          ) : (
            <div className="text-sm text-[#94a3b8]">{pseudoLabel}</div>
          )}
        </div>
      )}

      {/* Editor */}
      <div className="relative bg-[#0c0e14] border border-[#252836] rounded-xl focus-within:border-[#4f7af8]/60 transition-colors">
        {/* Character count */}
        <div className="absolute bottom-3 right-3 text-[10px] text-[#4a5068] pointer-events-none">
          {value.length} 字
        </div>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="输入标注文本…"
          className="w-full bg-transparent text-[#e2e8f0] text-base leading-relaxed p-4 pr-16 resize-none focus:outline-none min-h-[120px]"
          style={{ caretColor: '#4f7af8' }}
        />
      </div>

      {/* Model label (always visible summary below) */}
      <div className="flex gap-2 items-start">
        <span className="text-[10px] text-[#4a5068] mt-0.5 shrink-0">模型输出</span>
        <span className="text-xs text-[#4a5068] leading-relaxed line-clamp-2">{pseudoLabel}</span>
      </div>
    </div>
  );
}
