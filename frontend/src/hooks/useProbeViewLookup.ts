/**
 * Names for a set of probe view ids, read with ONE `listDatasets` call for the whole set — never
 * one request per line or per card (operator request 2026-11-27: every set line shows the view's
 * name AND its `pmd_…` id, and the lookups are batched).
 *
 * - The list is read again only when an id appears that the last read did not have (a view made
 *   since); ids already known, or a set that shrank, make no request.
 * - `pending` is true while the read for the ids now asked about is in flight, so a line can say
 *   "(looking up name…)" instead of a false "(name not found)".
 * - A failed read keeps the names already known; the ids it could not name stay unnamed, and the
 *   caller says so (`setLine`).
 * - No ids, no request.
 */
import { useEffect, useRef, useState } from 'react';

import { probeMonitorsApi } from '../api/probeMonitors';
import type { ProbeDataset } from '../types/probeMonitor';

export interface ProbeViewLookup {
  views: Record<string, ProbeDataset>;
  pending: boolean;
}

export function useProbeViewLookup(ids: readonly (string | null | undefined)[]): ProbeViewLookup {
  const [views, setViews] = useState<Record<string, ProbeDataset>>({});
  const viewsRef = useRef<Record<string, ProbeDataset>>({});
  const [loadedKey, setLoadedKey] = useState('');
  const key = [...new Set(ids.filter((id): id is string => typeof id === 'string' && id !== ''))]
    .sort()
    .join('|');

  useEffect(() => {
    if (!key) return;
    let cancelled = false;
    if (key.split('|').every((id) => id in viewsRef.current)) {
      setLoadedKey(key);
      return;
    }
    // Inside the chain, so even a lookup that throws before it returns a promise reads as failed.
    void Promise.resolve()
      .then(() => probeMonitorsApi.listDatasets())
      .then((list) => Object.fromEntries((list ?? []).map((v) => [v.id, v])) as Record<string, ProbeDataset>)
      .catch(() => null)
      .then((found) => {
        if (cancelled) return;
        if (found) {
          viewsRef.current = found;
          setViews(found);
        }
        setLoadedKey(key);
      });
    return () => {
      cancelled = true;
    };
  }, [key]);

  return { views, pending: key !== '' && loadedKey !== key };
}
