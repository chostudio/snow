import type { CoachReport, Session, TrickEvent } from '../types'
import { f1, f2, signed, timecode } from '../lib/format'
import { SEVERITY_LABEL } from '../lib/format'
import Sparkline from './Sparkline'
import SeverityIcon, { SEVERITY_COLOR } from './SeverityIcon'

interface Props {
  session: Session
  report: CoachReport
  event: TrickEvent | null
  onBack: () => void
}

/** The hand-off artefact: one page an athlete, coach or judge can keep.
 * Printing it is the PDF export — no server-side renderer to keep alive. */
export default function ReportView({ session, report, event, onBack }: Props) {
  const m = event?.metrics
  const i0 = event ? session.telemetry.f.findIndex((f) => f >= event.start_f) : 0
  const i1 = event ? session.telemetry.f.findIndex((f) => f > event.end_f) : -1
  const rpmSlice = session.telemetry.rpm.slice(i0, i1 === -1 ? undefined : i1)

  return (
    <div className="report">
      <div className="no-print" style={{ display: 'flex', gap: 8, marginBottom: 20 }}>
        <button className="btn" onClick={onBack}>← Back to review</button>
        <button className="btn btn-primary" onClick={() => window.print()}>Print / Save as PDF</button>
      </div>

      <div className="report-head">
        <div>
          <h1>Trick breakdown</h1>
          <div style={{ color: 'var(--ink-muted)', fontSize: 13 }}>
            Clip <strong>{session.name}</strong> · {f1(session.clip.duration)}s at {session.clip.fps} fps
            {session.clip.slowmo_factor !== 1 && ` · conformed slow motion ×${session.clip.slowmo_factor}`}
          </div>
        </div>
        <div style={{ marginLeft: 'auto', textAlign: 'right', fontSize: 12, color: 'var(--ink-muted)' }}>
          SNOW VAR<br />
          {new Date().toISOString().slice(0, 10)}
        </div>
      </div>

      {m && event ? (
        <>
          <div className="report-hero">
            <div>
              <div className="n">{m.nearest_spin_deg}°</div>
              <div className="l">called spin · {m.direction}</div>
            </div>
            <div>
              <div className="n">{signed(m.rotation_deg)}°</div>
              <div className="l">measured body-axis rotation</div>
            </div>
            <div>
              <div className="n">{Math.abs(m.peak_rpm).toFixed(0)}</div>
              <div className="l">peak RPM</div>
            </div>
            <div>
              <div className="n">{f2(m.air_time_s)}s</div>
              <div className="l">air time</div>
            </div>
            <div>
              <div className="n">{f1(m.max_axis_tilt_deg)}°</div>
              <div className="l">max axis tilt</div>
            </div>
          </div>

          <h2>Rotation rate through the air</h2>
          <Sparkline values={rpmSlice} color="var(--series-1)" width={780} height={110} zeroLine />
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--ink-muted)' }}>
            <span>{timecode(event.t_start)}</span>
            <span>peak {signed(m.peak_rpm)} RPM</span>
            <span>{timecode(event.t_end)}</span>
          </div>

          <h2>Measurements</h2>
          <table className="kv">
            <tbody>
              <tr><td>Frames</td><td>{event.start_f}–{event.end_f} ({event.end_f - event.start_f + 1})</td></tr>
              <tr><td>Rotation ({session.source.rotation_measure})</td><td>{signed(m.rotation_deg, 1)}°</td></tr>
              <tr><td>Rotation decay after peak</td><td>{signed(m.rotation_decay_dps2, 1)} °/s²</td></tr>
              <tr>
                <td>
                  Ankle rise above the pre-takeoff baseline
                  <br />
                  <span style={{ fontSize: 11 }}>camera-relative — the camera also moves, so this is not a pop height</span>
                </td>
                <td>{f2(m.pop_height_m)} m</td>
              </tr>
              <tr><td>Minimum tuck radius (wrists to spine)</td><td>{f2(m.min_tuck_radius_m)} m</td></tr>
              <tr><td>Knee flexion at touchdown</td><td>{f1(m.landing_knee_flex_deg)}°</td></tr>
              <tr><td>Mean joint confidence over the air</td><td>{(m.mean_confidence * 100).toFixed(0)}%</td></tr>
            </tbody>
          </table>
        </>
      ) : (
        <p className="note">No air event selected.</p>
      )}

      <h2>Findings</h2>
      {report.findings.length === 0 && <p className="note">No findings recorded.</p>}
      {report.findings.map((f, i) => (
        <div className="finding" key={i}>
          <span className="finding-icon"><SeverityIcon severity={f.severity} /></span>
          <div>
            <div className="sev-label" style={{ color: SEVERITY_COLOR[f.severity] }}>{SEVERITY_LABEL[f.severity]}</div>
            <div className="finding-title">{f.title}</div>
            <div className="finding-detail">{f.detail}</div>
          </div>
        </div>
      ))}

      <h2>Coaching cues</h2>
      <ul className="cues">
        {report.cues.map((c, i) => (
          <li className="cue" key={i}>
            <span className="cue-n">{i + 1}</span>
            <span>{c}</span>
          </li>
        ))}
      </ul>

      <h2>How these numbers were produced</h2>
      <p className="note" style={{ lineHeight: 1.6 }}>
        3D pose from the <strong>NVIDIA 3D Body Pose NIM</strong> ({session.skeleton.num_joints}-joint Nova
        skeleton), on {session.clip.tracked_frames} of {session.clip.total_frames} frames; the remaining frames
        are broadcast camera cuts where no rider is tracked and nothing is measured. Rotation is{' '}
        <strong>{session.source.rotation_measure}</strong>, integrated frame to frame and never across a cut.
        {session.source.metrics === 'snow-features'
          ? ' The camera\u2019s own rotation was estimated from the background and subtracted first, so the figure is the rider\u2019s turn and not the broadcast pan.'
          : ' The camera\u2019s pan has not been subtracted, so the absolute total is indicative only.'}{' '}
        {session.clip.slowmo_factor !== 1
          ? ` Rates are divided by the clip’s slow-motion factor (×${session.clip.slowmo_factor}) and are real-time.`
          : ` Rates and air times are in the clip’s own timebase (${session.clip.fps} fps); no slow-motion factor was given, so on slow-motion footage every rate is low by that factor and every air time high by it, while totals and angles are unaffected.`}{' '}
        Metric scale comes from an assumed pinhole focal length and is{' '}
        <strong>uncalibrated</strong>: angles and ratios are trustworthy, absolute metres are not.
        Kinematics source: <code>{session.source.metrics}</code>. Reasoning source:{' '}
        <code>{report.source}</code>.
      </p>
    </div>
  )
}
