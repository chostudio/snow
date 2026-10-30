import type { CoachReport, Session, SessionSummary } from './types'

/** Where session bundles come from.
 *
 * Unset (the default) reads the static bundles under `public/sessions/`, which
 * `tools/export_session.py` writes. Point `VITE_API_BASE` at `snow serve`
 * (src/snow/serve.py) and nothing else in the UI changes: it serves the same
 * routes plus `POST /<name>/coach`, the live Nemotron coach.
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

/** Ask the Nemotron coach (`snow serve`) for a fresh read on one event.
 *
 * With no coach endpoint (the static build, or the dev server alone) this
 * returns the report already embedded in the bundle. When the endpoint exists
 * but the model fails or returns an invalid report, it throws with the
 * server's reason: the panel shows that error, never a template in its place.
 */
export async function processClip(name: string): Promise<{ status: string; message: string }> {
  const root = BASE.replace(/\/sessions\/?$/, '')
  const res = await fetch(`${root}/api/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export async function predictScore(
  trick: string,
  level: string = 'WC',
  gender: string = 'M',
  final: boolean = true
): Promise<{ trick: string; score_if_landed: number; p_land: number; expected_score: number }> {
  const root = BASE.replace(/\/sessions\/?$/, '')
  const res = await fetch(`${root}/api/score`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ trick, level, gender, final }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

export async function fetchCoachReport(session: Session, eventId?: string): Promise<CoachReport> {
  let res: Response
  try {
    res = await fetch(`${BASE}/${session.name}/coach`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ event_id: eventId ?? null }),
    })
  } catch {
    return session.coach // nothing listening: static bundles only
  }
  const json = (res.headers.get('Content-Type') ?? '').includes('application/json')
  // A static host answers 404/405, or an HTML page, for a route it does not have.
  if (!json || res.status === 404 || res.status === 405) return session.coach
  const body = (await res.json()) as CoachReport | { error: string }
  if (!res.ok || 'error' in body) {
    throw new Error('error' in body ? body.error : `${res.status} ${res.statusText}`)
  }
  return body
}
