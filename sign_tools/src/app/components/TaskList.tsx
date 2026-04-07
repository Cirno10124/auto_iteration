import { FilterStatus, TaskItem, TaskStatus } from '../types';
import { SCENE_COLORS } from '../data/mockData';
import { CheckCircle2, SkipForward, Clock, AlertCircle } from 'lucide-react';

interface TaskListProps {
  tasks: TaskItem[];
  currentId: string;
  onSelect: (id: string) => void;
  filterStatus: FilterStatus;
}

const STATUS_ICONS: Record<TaskStatus, React.ReactNode> = {
  pending: <Clock size={12} className="text-[#4a5068]" />,
  accepted: <CheckCircle2 size={12} className="text-[#22c55e]" />,
  skipped: <SkipForward size={12} className="text-[#f59e0b]" />,
  review: <AlertCircle size={12} className="text-[#a78bfa]" />,
};

const STATUS_LABELS: Record<TaskStatus, string> = {
  pending: '待标注',
  accepted: '已接受',
  skipped: '已跳过',
  review: '待复核',
};

const GRADE_COLORS: Record<string, string> = {
  A: 'text-[#a78bfa] border-[#a78bfa]/40 bg-[#a78bfa]/10',
  B: 'text-[#60a5fa] border-[#60a5fa]/40 bg-[#60a5fa]/10',
  C: 'text-[#34d399] border-[#34d399]/40 bg-[#34d399]/10',
};

function ConfidenceBar({ value }: { value: number }) {
  const color =
    value < 0.4 ? '#ef4444' : value < 0.65 ? '#f59e0b' : '#22c55e';
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-12 h-1 bg-[#252836] rounded-full overflow-hidden">
        <div
          className="h-full rounded-full"
          style={{ width: `${value * 100}%`, backgroundColor: color }}
        />
      </div>
      <span className="text-[10px]" style={{ color }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

function formatDuration(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function TaskList({ tasks, currentId, onSelect, filterStatus }: TaskListProps) {
  const filtered =
    filterStatus === 'all' ? tasks : tasks.filter((t) => t.status === filterStatus);

  return (
    <aside className="w-[260px] bg-[#0c0e14] border-r border-[#252836] flex flex-col shrink-0">
      {/* Header */}
      <div className="px-3 py-2.5 border-b border-[#252836]">
        <div className="text-xs text-[#94a3b8]">
          任务队列
          <span className="ml-2 text-[#4a5068]">({filtered.length} 条)</span>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="flex items-center justify-center h-32 text-[#4a5068] text-sm">
            暂无任务
          </div>
        ) : (
          filtered.map((task) => {
            const isActive = task.id === currentId;
            return (
              <button
                key={task.id}
                onClick={() => onSelect(task.id)}
                className={`w-full text-left px-3 py-2.5 border-b border-[#1a1d2a] transition-all hover:bg-[#161820] ${
                  isActive ? 'bg-[#1b1e2b] border-l-2 border-l-[#4f7af8]' : ''
                }`}
              >
                {/* Row 1: filename + status */}
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span
                    className={`text-xs truncate max-w-[140px] ${
                      isActive ? 'text-[#e2e8f0]' : 'text-[#94a3b8]'
                    }`}
                  >
                    {task.audioFilename}
                  </span>
                  <div className="flex items-center gap-1 shrink-0">
                    {STATUS_ICONS[task.status]}
                  </div>
                </div>

                {/* Row 2: confidence + duration + grade */}
                <div className="flex items-center gap-2">
                  <ConfidenceBar value={task.confidence} />
                  <span className="text-[10px] text-[#4a5068]">
                    {formatDuration(task.duration)}
                  </span>
                  {task.labelGrade && (
                    <span
                      className={`text-[10px] border px-1 rounded ${GRADE_COLORS[task.labelGrade]}`}
                    >
                      {task.labelGrade}
                    </span>
                  )}
                </div>

                {/* Row 3: scene tag */}
                {task.scene && (
                  <div className="mt-1.5">
                    <span
                      className="text-[10px] px-1.5 py-0.5 rounded"
                      style={{
                        color: SCENE_COLORS[task.scene] ?? '#94a3b8',
                        backgroundColor: `${SCENE_COLORS[task.scene] ?? '#94a3b8'}18`,
                      }}
                    >
                      {task.scene}
                    </span>
                  </div>
                )}
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
