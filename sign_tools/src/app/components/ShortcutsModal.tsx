import { X } from 'lucide-react';

interface ShortcutsModalProps {
  onClose: () => void;
}

const SHORTCUTS = [
  { key: '空格', action: '播放 / 暂停音频', category: '播放' },
  { key: '← / J', action: '后退 3 秒', category: '播放' },
  { key: '→ / L', action: '前进 3 秒', category: '播放' },
  { key: 'Enter', action: '保存当前并进入下一条', category: '标注' },
  { key: 'S', action: '跳过当前条目', category: '标注' },
  { key: '1', action: '设置标签等级 A', category: '标注' },
  { key: '2', action: '设置标签等级 B', category: '标注' },
  { key: '3', action: '设置标签等级 C', category: '标注' },
  { key: 'Ctrl+Z / ⌘Z', action: '文本撤销（编辑区内）', category: '编辑' },
  { key: 'Esc', action: '取消编辑 / 关闭弹窗', category: '编辑' },
  { key: '?', action: '打开快捷键说明', category: '其他' },
];

const CATEGORIES = ['播放', '标注', '编辑', '其他'];

export function ShortcutsModal({ onClose }: ShortcutsModalProps) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="bg-[#13161f] border border-[#252836] rounded-2xl w-[520px] shadow-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-[#252836]">
          <div>
            <h2 className="text-[#e2e8f0] text-base">快捷键说明</h2>
            <p className="text-xs text-[#4a5068] mt-0.5">支持键盘快速操作，提升标注效率</p>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center text-[#4a5068] hover:text-[#e2e8f0] hover:bg-[#252836] rounded-lg transition-all"
          >
            <X size={16} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {CATEGORIES.map((cat) => {
            const items = SHORTCUTS.filter((s) => s.category === cat);
            return (
              <div key={cat} className="mb-5 last:mb-0">
                <div className="text-[10px] text-[#4a5068] uppercase tracking-widest mb-2">{cat}</div>
                <div className="space-y-1">
                  {items.map((s) => (
                    <div
                      key={s.key}
                      className="flex items-center justify-between py-1.5 px-3 rounded-lg hover:bg-[#1b1e2b] transition-all"
                    >
                      <span className="text-sm text-[#94a3b8]">{s.action}</span>
                      <div className="flex items-center gap-1">
                        {s.key.split(' / ').map((k) => (
                          <kbd
                            key={k}
                            className="text-xs bg-[#0c0e14] border border-[#252836] text-[#e2e8f0] px-2 py-1 rounded-md font-mono shadow"
                          >
                            {k}
                          </kbd>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-[#252836] flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-[#252836] hover:bg-[#2e3147] text-[#e2e8f0] text-sm rounded-lg transition-all"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}
