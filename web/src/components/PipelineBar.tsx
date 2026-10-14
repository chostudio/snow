import type { Session } from '../types'

interface Stage {
  label: string
  state: 'live' | 'stub'
  title: string
}

/** Honest provenance, in the header: which stages actually produced what is on
 * screen, and which are still stand-ins. */
export default function PipelineBar({ session }: { session: Session }) {
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
      label: 'Nemotron reasoning',
      state: session.source.coach === 'nemotron' ? 'live' : 'stub',
      title: 'Deep biomechanical reasoning over the event summary',
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
