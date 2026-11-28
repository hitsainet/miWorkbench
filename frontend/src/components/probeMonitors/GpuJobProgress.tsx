/**
 * A long probe GPU job — an evaluation or a WSP held-out capture — while it runs.
 *
 * ⚠ WHY THIS EXISTS (operator, 2026-11-27: "what is this job? where can I see its progress in a UI?"). A
 * probe evaluation over ten views ran for two hours and the only trace anywhere was a GPU-queue message
 * with no progress; this page showed each view's result only when that view finished. The server now
 * keeps a task-queue row for each job and publishes which view it is on and how many rows it has scored
 * (`progress_detail`); this renders that reading, here and in the Monitor's Active Operations.
 *
 * Read from `GET /task-queue/active`, so it survives a reload and shows a job started from MCP or another
 * tab. Every figure is the server's: a total still estimated says "about", speed and time left show only
 * once measured.
 */
import { ProgressBar } from '../common/ProgressBar';
import type { TaskQueueEntry } from '../../types/taskQueue';
import { describeViewRows, describeViewRowsDetail, PROBE_GPU_JOB_VERBS } from './gpuJobText';

/** One job's block: "Evaluating… view 3/10, 4,512/33,772 rows", its bar, and the view in hand. */
export function ProbeGpuJobProgress({ task }: { task: TaskQueueEntry }) {
  const verb = PROBE_GPU_JOB_VERBS[task.task_type] ?? 'Working';
  const detail = task.entity_info?.progress_detail;
  if (task.status === 'orphaned') {
    return (
      <p className="mt-2 text-xs text-red-300" data-testid="probe-gpu-job" data-task-type={task.task_type}>
        {verb} stopped reporting: the worker is presumed gone. Run it again.
      </p>
    );
  }
  if (task.status !== 'running') {
    return (
      <p className="mt-2 text-xs text-amber-300" data-testid="probe-gpu-job" data-task-type={task.task_type}>
        {verb === 'Evaluating' ? 'Evaluation' : 'Capture'} queued, waiting for a GPU
      </p>
    );
  }
  return (
    <div className="mt-2" data-testid="probe-gpu-job" data-task-type={task.task_type}>
      <p className="text-xs text-emerald-300" data-testid="probe-gpu-job-line">
        {verb}… {detail ? describeViewRows(detail) : 'starting'}
      </p>
      <ProgressBar progress={task.progress ?? 0} showPercentage={false} className="mt-1" />
      {detail ? (
        <p className="mt-0.5 text-[11px] text-slate-400" data-testid="probe-gpu-job-detail">
          {describeViewRowsDetail(detail)}
        </p>
      ) : null}
    </div>
  );
}
