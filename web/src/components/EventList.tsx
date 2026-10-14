import type { Session, TrickEvent } from '../types'
import { f2, signed, timecode } from '../lib/format'

interface Props {
  session: Session
  selectedId: string | null
  onSelect: (event: TrickEvent) => void
}

export default function EventList({ session, selectedId, onSelect }: Props) {
  if (session.events.length === 0) {
    return <p className="note">No air detected. The clip may be all transition riding, or the rider went untracked through the hits.</p>
  }
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {session.events.map((ev) => {
        const m = ev.metrics
        return (
          <button
            key={ev.id} className="event" aria-current={ev.id === selectedId}
            onClick={() => onSelect(ev)}
          >
            <div className="event-top">
              <span className="event-spin">{m.nearest_spin_deg}°</span>
              <span className="event-dir">{m.direction}</span>
              <span className="event-t">
                {timecode(ev.t_start)} → {timecode(ev.t_end)}
              </span>
            </div>
            <div className="event-grid">
              <div>
                <strong>{signed(m.rotation_deg)}°</strong>measured
              </div>
              <div>
                <strong>{signed(m.peak_rpm)}</strong>peak RPM
              </div>
              <div>
                <strong>{f2(m.air_time_s)}s</strong>air
              </div>
            </div>
          </button>
        )
      })}
    </div>
  )
}
