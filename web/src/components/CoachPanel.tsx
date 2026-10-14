import type { CoachReport, Session, TrickEvent } from '../types'
import { SEVERITY_LABEL } from '../lib/format'
import SeverityIcon, { SEVERITY_COLOR } from './SeverityIcon'

interface Props {
  session: Session
  report: CoachReport
  event: TrickEvent | null
  loading: boolean
  onRefresh: () => void
}

export default function CoachPanel({ session, report, event, loading, onRefresh }: Props) {
  const live = report.source === 'nemotron'
  return (
    <div>
      <div className="verdict">{report.verdict}</div>
      <div className="verdict-sub">
        {event
          ? `${event.id} · ${event.t_start.toFixed(2)}–${event.t_end.toFixed(2)}s · pose confidence ${(
              event.metrics.mean_confidence * 100
            ).toFixed(0)}%`
          : `${session.events.length} air${session.events.length === 1 ? '' : 's'} across the clip`}
      </div>

      {report.findings.length > 0 && (
        <div style={{ marginTop: 16 }}>
          {report.findings.map((f, i) => (
            <div className="finding" key={i}>
              <span className="finding-icon">
                <SeverityIcon severity={f.severity} />
              </span>
              <div>
                <div className="sev-label" style={{ color: SEVERITY_COLOR[f.severity] }}>
                  {SEVERITY_LABEL[f.severity]}
                </div>
                <div className="finding-title">{f.title}</div>
                <div className="finding-detail">{f.detail}</div>
              </div>
            </div>
          ))}
        </div>
      )}

      {report.cues.length > 0 && (
        <>
          <h4 style={{ margin: '18px 0 9px', fontSize: 12, letterSpacing: '0.06em', textTransform: 'uppercase', color: 'var(--ink-2)' }}>
            Coaching cues
          </h4>
          <ul className="cues">
            {report.cues.map((c, i) => (
              <li className="cue" key={i}>
                <span className="cue-n">{i + 1}</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </>
      )}

      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 18 }}>
        <span className="chip" data-state={live ? 'live' : 'stub'}>
          <span className="dot" />
          {live ? `Nemotron · ${report.model}` : 'Rule-based placeholder'}
        </span>
        <button className="btn btn-ghost" onClick={onRefresh} disabled={loading} style={{ marginLeft: 'auto' }}>
          {loading ? 'Asking…' : 'Re-run analysis'}
        </button>
      </div>
      {!live && (
        <p className="note" style={{ marginTop: 10 }}>
          Every sentence above is a template over a measured number, not model output.{' '}
          <strong>Re-run analysis</strong> posts the event's kinematics to{' '}
          <code style={{ fontFamily: 'var(--mono)' }}>POST /{session.name}/coach</code> and renders whatever the
          reasoning layer returns in this same shape.
        </p>
      )}
    </div>
  )
}
