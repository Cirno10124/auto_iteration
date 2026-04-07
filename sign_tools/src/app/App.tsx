import { useCallback, useEffect, useRef, useState } from 'react';
import { MOCK_TASKS } from './data/mockData';
import { FilterStatus, IssueTag, LabelGrade, SortMode, TaskItem } from './types';
import { TopBar } from './components/TopBar';
import { TaskList } from './components/TaskList';
import { AudioPlayer } from './components/AudioPlayer';
import { TextEditor } from './components/TextEditor';
import { RightPanel } from './components/RightPanel';
import { ActionBar } from './components/ActionBar';
import { ShortcutsModal } from './components/ShortcutsModal';
import { ImportExportModal } from './components/ImportExportModal';

function sortTasks(tasks: TaskItem[], mode: SortMode): TaskItem[] {
  const copy = [...tasks];
  switch (mode) {
    case 'confidence_asc':
      return copy.sort((a, b) => a.confidence - b.confidence);
    case 'confidence_mid':
      return copy.sort(
        (a, b) =>
          Math.abs(a.confidence - 0.5) - Math.abs(b.confidence - 0.5)
      );
    case 'sequential':
    default:
      return copy;
  }
}

export default function App() {
  const [tasks, setTasks] = useState<TaskItem[]>(MOCK_TASKS);
  const [currentId, setCurrentId] = useState<string>(MOCK_TASKS[0].id);
  const [filterStatus, setFilterStatus] = useState<FilterStatus>('all');
  const [sortMode, setSortMode] = useState<SortMode>('confidence_asc');
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [modalMode, setModalMode] = useState<'import' | 'export' | null>(null);

  // Audio player state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);

  // Per-task edits (grade, issue, text)
  const [draftGrade, setDraftGrade] = useState<LabelGrade | undefined>(undefined);
  const [draftIssue, setDraftIssue] = useState<IssueTag | undefined>(undefined);
  const [draftText, setDraftText] = useState('');

  const startTimeRef = useRef<number>(Date.now());
  const sortedTasks = sortTasks(tasks, sortMode);
  const currentTask = tasks.find((t) => t.id === currentId) ?? tasks[0];
  const currentSortedIndex = sortedTasks.findIndex((t) => t.id === currentId);

  // Init draft when task changes
  useEffect(() => {
    if (!currentTask) return;
    setDraftText(currentTask.finalText);
    setDraftGrade(currentTask.labelGrade);
    setDraftIssue(currentTask.issueTag);
    setCurrentTime(0);
    setIsPlaying(false);
    startTimeRef.current = Date.now();
  }, [currentId]); // eslint-disable-line react-hooks/exhaustive-deps

  const updateTask = useCallback(
    (id: string, patch: Partial<TaskItem>) => {
      setTasks((prev) =>
        prev.map((t) => (t.id === id ? { ...t, ...patch } : t))
      );
    },
    []
  );

  const advanceToNext = useCallback(() => {
    const pendingTasks = sortedTasks.filter(
      (t) => t.id !== currentId && t.status === 'pending'
    );
    if (pendingTasks.length > 0) {
      setCurrentId(pendingTasks[0].id);
    } else {
      // Go to next in sorted order
      const nextIndex = currentSortedIndex + 1;
      if (nextIndex < sortedTasks.length) {
        setCurrentId(sortedTasks[nextIndex].id);
      }
    }
  }, [sortedTasks, currentId, currentSortedIndex]);

  const handleSubmitNext = useCallback(() => {
    const elapsed = Math.round((Date.now() - startTimeRef.current) / 1000);
    updateTask(currentId, {
      status: 'accepted',
      finalText: draftText,
      labelGrade: draftGrade,
      issueTag: draftIssue,
      annotatedAt: Date.now(),
      timeSpent: elapsed,
    });
    advanceToNext();
  }, [currentId, draftText, draftGrade, draftIssue, updateTask, advanceToNext]);

  const handleAccept = useCallback(() => {
    const elapsed = Math.round((Date.now() - startTimeRef.current) / 1000);
    updateTask(currentId, {
      status: 'accepted',
      finalText: draftText,
      labelGrade: draftGrade,
      issueTag: draftIssue,
      annotatedAt: Date.now(),
      timeSpent: elapsed,
    });
  }, [currentId, draftText, draftGrade, draftIssue, updateTask]);

  const handleSkip = useCallback(() => {
    updateTask(currentId, {
      status: 'skipped',
      issueTag: draftIssue,
    });
    advanceToNext();
  }, [currentId, draftIssue, updateTask, advanceToNext]);

  const handleMarkReview = useCallback(() => {
    updateTask(currentId, {
      status: 'review',
      finalText: draftText,
      labelGrade: draftGrade,
      issueTag: draftIssue,
    });
    advanceToNext();
  }, [currentId, draftText, draftGrade, draftIssue, updateTask, advanceToNext]);

  const handleSeekDelta = useCallback(
    (delta: number) => {
      setCurrentTime((t) =>
        Math.max(0, Math.min(currentTask.duration, t + delta))
      );
    },
    [currentTask]
  );

  // Keyboard shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      const isInTextarea = tag === 'TEXTAREA' || tag === 'INPUT';

      // Global shortcuts (work everywhere)
      if (e.key === '?') { setShowShortcuts(true); return; }
      if (e.key === 'Escape') { setShowShortcuts(false); setModalMode(null); return; }

      // Audio shortcuts (not in textarea)
      if (!isInTextarea) {
        if (e.key === ' ') {
          e.preventDefault();
          setIsPlaying((p) => !p);
          return;
        }
        if (e.key === 'ArrowLeft' || e.key === 'j' || e.key === 'J') {
          handleSeekDelta(-3);
          return;
        }
        if (e.key === 'ArrowRight' || e.key === 'l' || e.key === 'L') {
          handleSeekDelta(3);
          return;
        }
        if (e.key === 's' || e.key === 'S') { handleSkip(); return; }
        if (e.key === 'Enter') { handleSubmitNext(); return; }
        if (e.key === '1') { setDraftGrade((g) => g === 'A' ? undefined : 'A'); return; }
        if (e.key === '2') { setDraftGrade((g) => g === 'B' ? undefined : 'B'); return; }
        if (e.key === '3') { setDraftGrade((g) => g === 'C' ? undefined : 'C'); return; }
      }

      // In textarea: Enter = submit
      if (isInTextarea && e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        handleSubmitNext();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [handleSeekDelta, handleSkip, handleSubmitNext]);

  // Stats
  const accepted = tasks.filter((t) => t.status === 'accepted').length;
  const skipped = tasks.filter((t) => t.status === 'skipped').length;
  const review = tasks.filter((t) => t.status === 'review').length;
  const pending = tasks.filter((t) => t.status === 'pending').length;
  const annotatedTasks = tasks.filter((t) => t.timeSpent !== undefined);
  const avgTime =
    annotatedTasks.length > 0
      ? Math.round(annotatedTasks.reduce((s, t) => s + (t.timeSpent ?? 0), 0) / annotatedTasks.length)
      : 0;

  const stats = { total: tasks.length, accepted, skipped, reviewed: review, pending, avgTimePerItem: avgTime };
  const isLastItem = currentSortedIndex >= sortedTasks.length - 1;

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0c0e14] overflow-hidden">
      {/* Top bar */}
      <TopBar
        stats={stats}
        filterStatus={filterStatus}
        sortMode={sortMode}
        onFilterChange={setFilterStatus}
        onSortChange={setSortMode}
        onShortcutsOpen={() => setShowShortcuts(true)}
        onImport={() => setModalMode('import')}
        onExport={() => setModalMode('export')}
      />

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Task List */}
        <TaskList
          tasks={sortedTasks}
          currentId={currentId}
          onSelect={(id) => setCurrentId(id)}
          filterStatus={filterStatus}
        />

        {/* Center: Main working area */}
        <main className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-5">
            {/* Audio Player */}
            <AudioPlayer
              filename={currentTask.audioFilename}
              duration={currentTask.duration}
              isPlaying={isPlaying}
              onPlayPause={() => setIsPlaying((p) => !p)}
              onSeekDelta={handleSeekDelta}
              currentTime={currentTime}
              onTimeUpdate={setCurrentTime}
            />

            {/* Text Editor */}
            <div className="bg-[#13161f] border border-[#252836] rounded-xl p-4">
              <TextEditor
                pseudoLabel={currentTask.pseudoLabel}
                value={draftText}
                onChange={setDraftText}
                onReset={() => setDraftText(currentTask.pseudoLabel)}
              />
            </div>

            {/* Keyboard hint strip */}
            <div className="flex items-center gap-4 px-1">
              {[
                { key: '空格', hint: '播放/暂停' },
                { key: 'Enter', hint: '保存并下一条' },
                { key: 'S', hint: '跳过' },
                { key: '←/→', hint: '±3秒' },
                { key: '1/2/3', hint: 'A/B/C 等级' },
                { key: '?', hint: '更多快捷键' },
              ].map(({ key, hint }) => (
                <div key={key} className="flex items-center gap-1.5">
                  <kbd className="text-[10px] bg-[#1b1e2b] border border-[#252836] text-[#e2e8f0] px-1.5 py-0.5 rounded font-mono">
                    {key}
                  </kbd>
                  <span className="text-[10px] text-[#4a5068]">{hint}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Action bar */}
          <ActionBar
            onAccept={handleAccept}
            onSubmitNext={handleSubmitNext}
            onSkip={handleSkip}
            onMarkReview={handleMarkReview}
            currentIndex={currentSortedIndex}
            total={sortedTasks.length}
            isLastItem={isLastItem}
          />
        </main>

        {/* Right: Metadata & controls */}
        <RightPanel
          task={currentTask}
          labelGrade={draftGrade}
          issueTag={draftIssue}
          onGradeChange={setDraftGrade}
          onIssueChange={setDraftIssue}
          sessionStats={{ accepted, skipped, review, avgTime }}
        />
      </div>

      {/* Modals */}
      {showShortcuts && <ShortcutsModal onClose={() => setShowShortcuts(false)} />}
      {modalMode && (
        <ImportExportModal
          mode={modalMode}
          tasks={tasks}
          onClose={() => setModalMode(null)}
          onImport={(imported) => {
            setTasks(imported);
            setCurrentId(imported[0]?.id ?? '');
            setModalMode(null);
          }}
        />
      )}
    </div>
  );
}