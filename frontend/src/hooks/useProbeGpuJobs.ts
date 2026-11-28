/**
 * The long probe GPU jobs that are queued or running — evaluations and WSP held-out captures — by probe.
 *
 * Read from `GET /task-queue/active`, the same rows the Monitor's Active Operations shows, so a job started
 * from MCP, from another tab or before a reload is found (2026-11-27: an evaluation ran for two hours and
 * this page could not say it existed). Polled every 5 s while `enabled` and the tab is visible; the server
 * writes a reading at most every 5 s.
 *
 * `onFinished(probeId)` fires when a job that was listed for a probe is no longer listed, so the page can
 * reload that probe's results — the evaluations a finished job wrote.
 */
import { useEffect, useRef, useState } from 'react';

import { getActiveTasks } from '../api/taskQueue';
import type { TaskQueueEntry } from '../types/taskQueue';
import { PROBE_GPU_JOB_VERBS } from '../components/probeMonitors/gpuJobText';

export const PROBE_GPU_JOB_POLL_MS = 5000;

export function isProbeGpuJob(entry: TaskQueueEntry): boolean {
  return entry.entity_type === 'probe_monitor' && String(entry.task_type) in PROBE_GPU_JOB_VERBS;
}

export function useProbeGpuJobs(
  enabled: boolean,
  onFinished?: (probeId: string) => void,
): Record<string, TaskQueueEntry[]> {
  const [byProbe, setByProbe] = useState<Record<string, TaskQueueEntry[]>>({});
  const finished = useRef(onFinished);
  useEffect(() => {
    finished.current = onFinished;
  }, [onFinished]);
  const seen = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (!enabled) return undefined;
    let live = true;
    const tick = async () => {
      if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
      try {
        const res = await getActiveTasks();
        if (!live) return;
        const next: Record<string, TaskQueueEntry[]> = {};
        for (const entry of (res.data ?? []).filter(isProbeGpuJob)) {
          (next[entry.entity_id] ??= []).push(entry);
        }
        const now = new Set(Object.keys(next));
        for (const probeId of seen.current) {
          if (!now.has(probeId)) finished.current?.(probeId);
        }
        seen.current = now;
        setByProbe(next);
      } catch {
        // A failed poll keeps the last reading: "nothing is running" and "I could not ask" are different.
      }
    };
    void tick();
    const id = window.setInterval(tick, PROBE_GPU_JOB_POLL_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [enabled]);

  return byProbe;
}
