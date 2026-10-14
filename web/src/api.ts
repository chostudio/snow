import type { CoachReport, Session, SessionSummary } from './types'

/** Where session bundles come from.
 *
 * Unset (the default) reads the static bundles under `public/sessions/`, which
 * `tools/export_session.py` writes. Point `VITE_API_BASE` at the pipeline's own
 * HTTP surface once it exists and nothing else in the UI changes, as long as it
 * serves the same two routes plus, optionally, `POST /<name>/coach`.
 */
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? '/sessions'

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} — ${BASE}${path}`)
  return (await res.json()) as T
}

export async function fetchIndex(): Promise<SessionSummary[]> {
  const data = await getJSON<{ sessions: SessionSummary[] }>('/index.json')
  return data.sessions
}

export async function fetchSession(name: string): Promise<Session> {
  return getJSON<Session>(`/${name}/session.json`)
}

export function clipUrl(session: Session): string {
  return `${BASE}/${session.name}/${session.clip.src}`
}

/** Ask the reasoning layer for a fresh read on one event.
 *
 * Falls back to the report already embedded in the bundle, so the panel is
 * never empty while the Nemotron stage is still being wired up.
 */
export async function fetchCoachReport(session: Session, eventId?: string): Promise<CoachReport> {
  try {
    const res = await fetch(`${BASE}/${session.name}/coach`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_id: eventId ?? null }),
    })
    if (res.ok) return (await res.json()) as CoachReport
  } catch {
    /* no live endpoint — expected until the reasoning stage ships */
  }
  return session.coach
}
