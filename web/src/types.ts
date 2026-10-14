/** The session bundle contract, written by `tools/export_session.py`.
 *
 * Everything the UI renders comes from this shape. When the pipeline grows a
 * real HTTP surface, serving this same JSON from it is the whole integration —
 * see `src/api.ts`.
 */

export type MetricsSource = 'display-shim' | 'snow-features'
export type CoachSource = 'rule-based' | 'nemotron'
export type Severity = 'good' | 'warning' | 'serious' | 'critical'

export interface ClipInfo {
  src: string
  fps: number
  real_fps: number
  slowmo_factor: number
  frames: number
  total_frames: number
  tracked_frames: number
  width: number
  height: number
  duration: number
}

export interface EventMetrics {
  rotation_deg: number
  nearest_spin_deg: number
  direction: string
  peak_rpm: number
  air_time_s: number
  pop_height_m: number
  max_axis_tilt_deg: number
  min_tuck_radius_m: number
  landing_knee_flex_deg: number
  rotation_decay_dps2: number
  mean_confidence: number
}

export interface TrickEvent {
  id: string
  kind: string
  start_f: number
  end_f: number
  peak_f: number
  t_start: number
  t_end: number
  metrics: EventMetrics
}

export interface Telemetry {
  f: number[]
  t: number[]
  rotation_deg: number[]
  rpm: number[]
  axis_tilt_deg: number[]
  ankle_height_m: number[]
  tuck_radius_m: number[]
  knee_flex_deg: number[]
  confidence: number[]
  /** Present only when `snow features` supplied the rotation: how much of the
   * raw turn was the broadcast camera's own pan, already subtracted. */
  camera_rotation_cum_deg?: number[]
}

/** Per-frame pose. `k2` is source pixels, `k3` is millimetres in the NIM's
 * camera frame (X right, Y down, Z forward), `conf` is 0-100. */
export interface PoseFrame {
  f: number
  bbox: [number, number, number, number]
  k2: [number, number][]
  k3: [number, number, number][]
  conf: number[]
}

export interface Finding {
  severity: Severity
  title: string
  detail: string
}

export interface CoachReport {
  source: CoachSource
  model: string | null
  verdict: string
  headline_metric: { label: string; value: number; unit: string; event_id: string } | null
  findings: Finding[]
  cues: string[]
}

export interface Session {
  schema: number
  name: string
  clip: ClipInfo
  source: {
    metrics: MetricsSource
    /** Which rotation measure is on screen, in words. */
    rotation_measure: string
    coach: CoachSource
    pose: string
  }
  skeleton: {
    parents: number[]
    links: [number, number][]
    joints: Record<string, number>
    num_joints: number
  }
  /** Inclusive frame ranges with no tracked rider, i.e. camera cuts. */
  gaps: [number, number][]
  telemetry: Telemetry
  events: TrickEvent[]
  coach: CoachReport
  frames: PoseFrame[]
}

export interface SessionSummary {
  name: string
  duration: number
  events: number
  tracked_frames: number
}

export type TelemetryKey = Exclude<keyof Telemetry, 'f' | 't'>
