import type { Session, TrickEvent } from '../types'
import { f1, f2, signed } from '../lib/format'
import Sparkline from './Sparkline'
import SeverityIcon from './SeverityIcon'
import type { Severity } from '../types'

interface TileProps {
  label: string
  value: string
  unit?: string
  foot?: string
  severity?: Severity
  spark?: { values: number[]; color: string; zeroLine?: boolean }
}

function Tile({ label, value, unit, foot, severity, spark }: TileProps) {
  return (
    <div className="tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">
        {value}
        {unit && <span className="unit">{unit}</span>}
      </div>
      {spark && <Sparkline values={spark.values} color={spark.color} zeroLine={spark.zeroLine} />}
      {foot && (
        <div className="tile-foot">
          {severity && <SeverityIcon severity={severity} size={13} />}
          {foot}
        </div>
      )}
    </div>
  )
}

interface Props {
  session: Session
  event: TrickEvent | null
}

export default function MetricTiles({ session, event }: Props) {
  const { telemetry } = session

  const cam = telemetry.camera_rotation_cum_deg
  const cameraPan = cam && cam.length ? cam[cam.length - 1] : null

  if (!event) {
    const peak = telemetry.rpm.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0)
    return (
      <div className="tiles">
        <Tile
          label="Tracked frames" value={`${session.clip.tracked_frames}`}
          foot={`of ${session.clip.total_frames} · ${session.gaps.length} camera cuts`}
        />
        <Tile
          label="Peak rotation rate" value={signed(peak)} unit="RPM"
          spark={{ values: telemetry.rpm, color: 'var(--series-1)', zeroLine: true }}
        />
        <Tile label="Airs detected" value={`${session.events.length}`} foot="select one for a breakdown" />
        <Tile label="Clip length" value={f1(session.clip.duration)} unit="s" foot={`${session.clip.fps} fps`} />
        {cameraPan !== null && (
          <Tile
            label="Camera pan removed" value={f1(Math.abs(cameraPan))} unit="°"
            foot="subtracted before any rotation was counted"
            spark={{ values: telemetry.camera_rotation_cum_deg!, color: 'var(--ink-muted)' }}
          />
        )}
      </div>
    )
  }

  const m = event.metrics
  const i0 = telemetry.f.findIndex((f) => f >= event.start_f)
  const i1 = telemetry.f.findIndex((f) => f > event.end_f)
  const slice = <K extends 'rpm' | 'rotation_deg' | 'axis_tilt_deg' | 'knee_flex_deg'>(k: K) =>
    telemetry[k].slice(i0, i1 === -1 ? undefined : i1)

  const short = m.nearest_spin_deg - Math.abs(m.rotation_deg)
  const stiff = m.landing_knee_flex_deg > 150

  return (
    <div className="tiles">
      <Tile
        label={session.source.metrics === 'snow-features' ? 'Rotation · trick axis' : 'Rotation · body axis'}
        value={signed(m.rotation_deg)} unit="°"
        foot={
          Math.abs(short) < 15
            ? `clean ${m.nearest_spin_deg}°`
            : short > 0
              ? `${short.toFixed(0)}° short of ${m.nearest_spin_deg}°`
              : `${(-short).toFixed(0)}° past ${m.nearest_spin_deg}°`
        }
        severity={Math.abs(short) < 15 ? 'good' : 'warning'}
        spark={{ values: slice('rotation_deg'), color: 'var(--series-3)', zeroLine: true }}
      />
      <Tile
        label="Peak rotation rate" value={signed(m.peak_rpm)} unit="RPM"
        foot={`${(Math.abs(m.peak_rpm) * 6).toFixed(0)}°/s`}
        spark={{ values: slice('rpm'), color: 'var(--series-1)', zeroLine: true }}
      />
      <Tile label="Air time" value={f2(m.air_time_s)} unit="s" foot={`frames ${event.start_f}–${event.end_f}`} />
      <Tile
        label="Max axis tilt" value={f1(m.max_axis_tilt_deg)} unit="°"
        foot={m.max_axis_tilt_deg > 60 ? 'corked, off-axis' : 'flat spin'}
        severity={m.max_axis_tilt_deg > 60 ? 'good' : undefined}
        spark={{ values: slice('axis_tilt_deg'), color: 'var(--series-2)' }}
      />
      <Tile label="Tuck radius (min)" value={f2(m.min_tuck_radius_m)} unit="m" foot="wrists to spine" />
      <Tile
        label="Landing knee flex" value={f1(m.landing_knee_flex_deg)} unit="°"
        foot={stiff ? 'Risk · stiff on impact' : 'absorbing'}
        severity={stiff ? 'critical' : 'good'}
        spark={{ values: slice('knee_flex_deg'), color: 'var(--series-2)' }}
      />
    </div>
  )
}
