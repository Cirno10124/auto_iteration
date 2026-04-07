export type TaskStatus = 'pending' | 'accepted' | 'skipped' | 'review';
export type LabelGrade = 'A' | 'B' | 'C';
export type IssueTag =
  | 'unclear_audio'
  | 'wrong_language'
  | 'high_noise'
  | 'needs_review'
  | 'truncated'
  | 'other';

export interface TaskItem {
  id: string;
  audioFilename: string;
  audioPath: string;
  duration: number; // seconds
  confidence: number; // 0–1
  pseudoLabel: string; // original model output
  finalText: string; // annotator edited text
  status: TaskStatus;
  labelGrade?: LabelGrade;
  issueTag?: IssueTag;
  batchId: string;
  scene?: string;
  priority: number; // 1 = high, 3 = low
  annotatedAt?: number; // timestamp
  timeSpent?: number; // seconds
}

export interface SessionStats {
  total: number;
  accepted: number;
  skipped: number;
  reviewed: number;
  pending: number;
  avgTimePerItem: number;
}

export type SortMode = 'confidence_asc' | 'confidence_mid' | 'sequential';
export type FilterStatus = 'all' | 'pending' | 'accepted' | 'skipped' | 'review';
