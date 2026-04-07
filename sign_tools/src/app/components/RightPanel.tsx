import { IssueTag, LabelGrade, TaskItem } from '../types';
import { ISSUE_TAG_LABELS, SCENE_COLORS } from '../data/mockData';
import { Clock, FileAudio, Gauge, Tag, AlertTriangle } from 'lucide-react';

interface RightPanelProps {
  task: TaskItem;
  labelGrade: LabelGrade | undefined;
  issueTag: IssueTag | undefined;
  onGradeChange: (g: LabelGrade | undefined) => void;
  onIssueChange: (t: IssueTag | undefined) => void;
  sessionStats: { accepted: number; skipped: number; review: number; avgTime: number };
}

const GRADES: { value: LabelGrade; label: string; desc: string; color: string; bg: string; border: string }[] = [
  {
    value: 'A',
    label: 'A 级',
    desc: '严格人工审核',
    color: 'text-[#a78bfa]',
    bg: 'bg-[#a78bfa]/10',
    border: 'border-[#a78bfa]/50',
  },
  {
    value: 'B',
    label: 'B 级',
    desc: '快速过目',
    color: 'text-[#60a5fa]',
    bg: 'bg-[#60a5fa]/10',
    border: 'border-[#60a5fa]/50',
  },
  {
    value: 'C',
    label: 'C 级',
    desc: '模型输出未改',
    color: 'text-[#34d399]',
    bg: 'bg-[#34d399]/10',
    border: 'border-[#34d399]/50',
  },
];

const ISSUE_TAGS = Object.entries(ISSUE_TAG_LABELS) as [IssueTag, string][];

function InfoRow({ icon, label, value }: { icon: React.ReactNode; label: string; value: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-[#1a1d2a]">
      <div className="flex items-center gap-1.5 text-[#4a5068] text-xs">
        {icon}
        {label}
      </div>
      <div className="text-xs text-[#94a3b8]">{value}</div>
    </div>
  );
}

function formatDuration(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return m > 0 ? `${m}分${s}秒` : `${s}秒`;
}

export function RightPanel({
  task,
  labelGrade,
  issueTag,
  onGradeChange,
  onIssueChange,
  sessionStats,
}: RightPanelProps) {
  return (
    <aside className="w-[220px] bg-[#0c0e14] border-l border-[#252836] flex flex-col shrink-0 overflow-y-auto">
      {/* Section: Task Info */}
      <div className="px-3 py-2.5 border-b border-[#252836]">
        <div className="text-xs text-[#4a5068] mb-1">样本信息</div>
        <InfoRow
          icon={<FileAudio size={11} />}
          label="文件名"
          value={
            <span className="truncate max-w-[100px] block" title={task.audioFilename}>
              {task.audioFilename}
            </span>
          }
        />
        <InfoRow
          icon={<Clock size={11} />}
          label="时长"
          value={formatDuration(task.duration)}
        />
        <InfoRow
          icon={<Gauge size={11} />}
          label="置信度"
          value={
            <span
              className={
                task.confidence < 0.4
                  ? 'text-[#ef4444]'
                  : task.confidence < 0.65
                  ? 'text-[#f59e0b]'
                  : 'text-[#22c55e]'
              }
            >
              {Math.round(task.confidence * 100)}%
            </span>
          }
        />
        {task.scene && (
          <InfoRow
            icon={<Tag size={11} />}
            label="场景"
            value={
              <span
                style={{
                  color: SCENE_COLORS[task.scene] ?? '#94a3b8',
                }}
              >
                {task.scene}
              </span>
            }
          />
        )}
        <InfoRow
          icon={<Tag size={11} />}
          label="批次"
          value={task.batchId}
        />
      </div>

      {/* Section: Label Grade */}
      <div className="px-3 py-2.5 border-b border-[#252836]">
        <div className="text-xs text-[#4a5068] mb-2">标签等级 <span className="text-[10px]">(1/2/3)</span></div>
        <div className="flex flex-col gap-1.5">
          {GRADES.map((g) => (
            <button
              key={g.value}
              onClick={() => onGradeChange(labelGrade === g.value ? undefined : g.value)}
              className={`flex items-center justify-between px-2.5 py-2 rounded-lg border transition-all text-left ${
                labelGrade === g.value
                  ? `${g.bg} ${g.border} ${g.color}`
                  : 'border-[#252836] text-[#4a5068] hover:border-[#3a3d4a] hover:text-[#94a3b8]'
              }`}
            >
              <div>
                <div className="text-xs">{g.label}</div>
                <div className="text-[10px] opacity-70">{g.desc}</div>
              </div>
              {labelGrade === g.value && (
                <div className={`w-2 h-2 rounded-full ${g.bg.replace('/10', '')} border ${g.border}`} />
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Section: Issue Tags */}
      <div className="px-3 py-2.5 border-b border-[#252836]">
        <div className="flex items-center gap-1.5 text-xs text-[#4a5068] mb-2">
          <AlertTriangle size={11} />
          问题标记
        </div>
        <div className="flex flex-wrap gap-1.5">
          {ISSUE_TAGS.map(([tag, label]) => (
            <button
              key={tag}
              onClick={() => onIssueChange(issueTag === tag ? undefined : tag)}
              className={`text-[10px] px-2 py-1 rounded-lg border transition-all ${
                issueTag === tag
                  ? 'bg-[#ef4444]/15 border-[#ef4444]/40 text-[#ef4444]'
                  : 'border-[#252836] text-[#4a5068] hover:border-[#3a3d4a] hover:text-[#94a3b8]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Section: Session Stats */}
      <div className="px-3 py-2.5">
        <div className="text-xs text-[#4a5068] mb-2">本次统计</div>
        <div className="grid grid-cols-2 gap-2">
          <div className="bg-[#13161f] rounded-lg p-2.5 text-center">
            <div className="text-lg text-[#22c55e]">{sessionStats.accepted}</div>
            <div className="text-[10px] text-[#4a5068]">已接受</div>
          </div>
          <div className="bg-[#13161f] rounded-lg p-2.5 text-center">
            <div className="text-lg text-[#f59e0b]">{sessionStats.skipped}</div>
            <div className="text-[10px] text-[#4a5068]">已跳过</div>
          </div>
          <div className="bg-[#13161f] rounded-lg p-2.5 text-center">
            <div className="text-lg text-[#a78bfa]">{sessionStats.review}</div>
            <div className="text-[10px] text-[#4a5068]">待复核</div>
          </div>
          <div className="bg-[#13161f] rounded-lg p-2.5 text-center">
            <div className="text-lg text-[#60a5fa]">
              {sessionStats.avgTime > 0 ? `${sessionStats.avgTime}s` : '—'}
            </div>
            <div className="text-[10px] text-[#4a5068]">平均耗时</div>
          </div>
        </div>
      </div>
    </aside>
  );
}
