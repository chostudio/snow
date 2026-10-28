import type { CoachReport, Session } from '../types'

interface Stage {
  label: string
  state: 'live' | 'stub'
  title: string
}

/** Honest provenance, in the header: which stages actually produced what is on
 * screen, and which are still stand-ins. The coach chip follows the report on
 * screen, not the bundle: it is live only while a Nemotron report is shown. */
export default function PipelineBar({ session, report }: { session: Session; report: CoachReport }) {
  const coachLive = report.source === 'nemotron'
  const stages: Stage[] = [
    { label: 'YOLO + ByteTrack', state: 'live', title: 'Local detection and tracking, CPU/MPS' },
    { label: 'Body Pose NIM', state: 'live', title: 'NVIDIA 3D Body Pose NIM on NVCF — 77-joint 3D skeleton' },
    {
      label: session.source.metrics === 'snow-features' ? 'Camera-compensated kinematics' : 'Kinematics',
      state: session.source.metrics === 'snow-features' ? 'live' : 'stub',
      title:
        session.source.metrics === 'snow-features'
          ? `snow features — ${session.source.rotation_measure}, camera pan removed`
          : 'Display shim in tools/export_session.py — body-axis rotation, not camera-compensated',
    },
    {
      label: coachLive ? 'Nemotron coach' : 'Coach (template)',
      state: coachLive ? 'live' : 'stub',
      title: coachLive
        ? `${report.model} on the NVIDIA API, reasoning over the measured features only`
        : 'Rule-based template over measured numbers; run `snow serve` and Re-run analysis for Nemotron',
    },
  ]
  return (
    <div className="chips">
      {stages.map((s, i) => (
        <span key={s.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          {i > 0 && <span className="chip-arrow">→</span>}
          <span className="chip" data-state={s.state} title={s.title}>
            <span className="dot" />
            {s.label}
          </span>
        </span>
      ))}
    </div>
  )
}
