import { useEffect, useRef, useState, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  RotateCw,
  Volume2,
  Volume1,
  VolumeX,
} from 'lucide-react';

interface AudioPlayerProps {
  filename: string;
  duration: number; // total simulated duration in seconds
  isPlaying: boolean;
  onPlayPause: () => void;
  onSeekDelta: (delta: number) => void;
  currentTime: number;
  onTimeUpdate: (t: number) => void;
}

const SPEED_OPTIONS = [0.5, 0.75, 1, 1.25, 1.5, 2];

function formatTime(sec: number) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

// Simple fake waveform bars
function WaveformBar({ active, height }: { active: boolean; height: number }) {
  return (
    <div
      className="rounded-full transition-colors duration-150"
      style={{
        width: 3,
        height,
        backgroundColor: active ? '#4f7af8' : '#252836',
      }}
    />
  );
}

const WAVEFORM_BARS = Array.from({ length: 80 }, () =>
  Math.max(4, Math.floor(Math.random() * 32))
);

export function AudioPlayer({
  filename,
  duration,
  isPlaying,
  onPlayPause,
  onSeekDelta,
  currentTime,
  onTimeUpdate,
}: AudioPlayerProps) {
  const [speed, setSpeed] = useState(1);
  const [volume, setVolume] = useState(0.8);
  const [showVolume, setShowVolume] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const progressRef = useRef<HTMLDivElement>(null);
  const currentTimeRef = useRef(currentTime);

  // Keep ref in sync
  useEffect(() => {
    currentTimeRef.current = currentTime;
  }, [currentTime]);

  // Simulate playback
  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        const next = Math.min(currentTimeRef.current + 0.1 * speed, duration);
        currentTimeRef.current = next;
        onTimeUpdate(next);
        if (next >= duration) {
          clearInterval(intervalRef.current!);
        }
      }, 100);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isPlaying, speed, duration]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleProgressClick = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      if (!progressRef.current) return;
      const rect = progressRef.current.getBoundingClientRect();
      const ratio = (e.clientX - rect.left) / rect.width;
      onTimeUpdate(Math.max(0, Math.min(duration, ratio * duration)));
    },
    [duration, onTimeUpdate]
  );

  const pct = duration > 0 ? (currentTime / duration) * 100 : 0;
  const activeBarIndex = Math.floor((currentTime / duration) * WAVEFORM_BARS.length);

  const VolumeIcon = volume === 0 ? VolumeX : volume < 0.5 ? Volume1 : Volume2;

  return (
    <div className="bg-[#13161f] border border-[#252836] rounded-xl p-4 flex flex-col gap-3">
      {/* Filename */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#4f7af8] animate-pulse" />
          <span className="text-sm text-[#e2e8f0] truncate max-w-[280px]">{filename}</span>
        </div>
        {/* Speed control */}
        <div className="flex items-center gap-1">
          {SPEED_OPTIONS.map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`text-xs px-2 py-1 rounded transition-all ${
                speed === s
                  ? 'bg-[#4f7af8] text-white'
                  : 'text-[#4a5068] hover:text-[#94a3b8] hover:bg-[#252836]'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>
      </div>

      {/* Waveform */}
      <div className="flex items-center gap-px h-10 cursor-pointer" onClick={handleProgressClick}>
        {WAVEFORM_BARS.map((h, i) => (
          <WaveformBar key={i} height={h} active={i <= activeBarIndex} />
        ))}
      </div>

      {/* Progress bar */}
      <div
        ref={progressRef}
        className="relative h-1.5 bg-[#252836] rounded-full cursor-pointer group"
        onClick={handleProgressClick}
      >
        <div
          className="absolute inset-y-0 left-0 bg-gradient-to-r from-[#4f7af8] to-[#6b8ffa] rounded-full transition-all"
          style={{ width: `${pct}%` }}
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg opacity-0 group-hover:opacity-100 transition-opacity border-2 border-[#4f7af8]"
          style={{ left: `calc(${pct}% - 6px)` }}
        />
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-3">
        {/* Seek back */}
        <button
          onClick={() => onSeekDelta(-3)}
          title="-3s (← / J)"
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#252836] transition-all text-xs"
        >
          <RotateCcw size={14} />
          <span>3s</span>
        </button>

        {/* Play/Pause */}
        <button
          onClick={onPlayPause}
          title="播放/暂停 (空格)"
          className="w-10 h-10 rounded-full bg-[#4f7af8] hover:bg-[#6b8ffa] flex items-center justify-center text-white transition-all shadow-lg shadow-[#4f7af8]/20 active:scale-95"
        >
          {isPlaying ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
        </button>

        {/* Seek forward */}
        <button
          onClick={() => onSeekDelta(3)}
          title="+3s (→ / L)"
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[#94a3b8] hover:text-[#e2e8f0] hover:bg-[#252836] transition-all text-xs"
        >
          <span>3s</span>
          <RotateCw size={14} />
        </button>

        {/* Time display */}
        <div className="flex-1 text-center text-sm text-[#94a3b8] font-mono">
          <span className="text-[#e2e8f0]">{formatTime(currentTime)}</span>
          <span className="mx-1 text-[#4a5068]">/</span>
          <span>{formatTime(duration)}</span>
        </div>

        {/* Volume */}
        <div className="relative flex items-center">
          <button
            onClick={() => setShowVolume(!showVolume)}
            className="w-8 h-8 flex items-center justify-center text-[#4a5068] hover:text-[#94a3b8] hover:bg-[#252836] rounded-lg transition-all"
          >
            <VolumeIcon size={16} />
          </button>
          {showVolume && (
            <div className="absolute bottom-10 right-0 bg-[#1b1e2b] border border-[#252836] rounded-xl p-3 shadow-xl z-20 flex flex-col items-center gap-2">
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(e) => setVolume(parseFloat(e.target.value))}
                className="h-20 appearance-none cursor-pointer"
                style={{ writingMode: 'vertical-lr', direction: 'rtl' }}
              />
              <span className="text-xs text-[#4a5068]">{Math.round(volume * 100)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}