import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from './api'
import type { CoachReport, Session, SessionSummary, TrickEvent } from './types'
import { coreLinks, frameIndex, telemetryIndex } from './lib/pose'
import { useVideoClock } from './lib/useVideoClock'
import { f1, timecode } from './lib/format'
import VideoStage, { type OverlayFlags } from './components/VideoStage'
import TelemetryTimeline from './components/TelemetryTimeline'
import MetricTiles from './components/MetricTiles'
import CoachPanel from './components/CoachPanel'
import EventList from './components/EventList'
import PipelineBar from './components/PipelineBar'
import PoseInspector3D from './components/PoseInspector3D'
import ReportView from './components/ReportView'

export default function App() {
  const [index, setIndex] = useState<SessionSummary[] | null>(null)
  const [name, setName] = useState<string | null>(null)
  const [session, setSession] = useState<Session | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api
      .fetchIndex()
      .then((list) => {
        setIndex(list)
        setName((cur) => cur ?? list[0]?.name ?? null)
      })
      .catch((e: Error) => setError(e.message))
  }, [])

  useEffect(() => {
    if (!name) return
    setSession(null)
    api.fetchSession(name).then(setSession).catch((e: Error) => setError(e.message))
  }, [name])

  if (error) {
    return (
      <div className="error card">
        <h2 style={{ marginBottom: 8 }}>No session bundle</h2>
        <p className="note">The UI reads bundles written by the exporter. Build one first:</p>
        <pre>cd web
../.venv/bin/python tools/export_session.py run1</pre>
        <p className="note" style={{ marginTop: 10 }}>
          That needs <code>data/poses/&lt;name&gt;.json</code> and <code>data/clips/&lt;name&gt;.mp4</code>, i.e. a
          finished <code>snow run</code>.
        </p>
        <pre style={{ marginTop: 12 }}>{error}</pre>
      </div>
    )
  }

  if (!session || !index) {
    return (
      <div className="loading">
        <div>Loading session…</div>
      </div>
    )
  }

  return <Review key={session.name} session={session} index={index} onPick={setName} />
}

function Review({
  session, index, onPick,
}: {
  session: Session
  index: SessionSummary[]
  onPick: (name: string) => void
}) {
  const { fps, total_frames } = session.clip
  const clock = useVideoClock(fps, total_frames)
  const { frame, seekFrame } = clock

  const poses = useMemo(() => frameIndex(session), [session])
  const links = useMemo(() => coreLinks(session), [session])
  const tIdx = useMemo(() => telemetryIndex(session), [session])
  const [selectedId, setSelectedId] = useState<string | null>(session.events[0]?.id ?? null)
  const [tab, setTab] = useState<'coach' | 'tricks' | 'pose'>('coach')
  const [flags, setFlags] = useState<OverlayFlags>({ skeleton: true, box: false, trail: true })
  const [report, setReport] = useState<CoachReport>(session.coach)
  const [coachLoading, setCoachLoading] = useState(false)
  const [showReport, setShowReport] = useState(false)

  const selected = session.events.find((e) => e.id === selectedId) ?? null
  const active = session.events.find((e) => frame >= e.start_f && frame <= e.end_f) ?? null
  const ti = tIdx.get(frame)
  const rpm = ti === undefined ? null : session.telemetry.rpm[ti]
  const rotation = ti === undefined ? null : session.telemetry.rotation_deg[ti]

  const flagged = useMemo(() => {
    const out = new Set<number>()
    if (ti !== undefined && session.telemetry.knee_flex_deg[ti] > 150) {
      out.add(session.skeleton.joints.knee_l)
      out.add(session.skeleton.joints.knee_r)
    }
    return out
  }, [ti, session])

  const pickEvent = useCallback(
    (ev: TrickEvent, seek: boolean) => {
      setSelectedId(ev.id)
      if (seek) seekFrame(ev.peak_f)
    },
    [seekFrame],
  )

  const refreshCoach = useCallback(() => {
    setCoachLoading(true)
    api
      .fetchCoachReport(session, selectedId ?? undefined)
      .then(setReport)
      .finally(() => setCoachLoading(false))
  }, [session, selectedId])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLElement && /INPUT|SELECT|TEXTAREA/.test(e.target.tagName)) return
      const step = e.shiftKey ? 10 : 1
      if (e.code === 'Space') {
        e.preventDefault()
        clock.toggle()
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault()
        clock.pause()
        seekFrame(frame - step)
      } else if (e.code === 'ArrowRight') {
        e.preventDefault()
        clock.pause()
        seekFrame(frame + step)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [clock, frame, seekFrame])

  if (showReport) {
    return <ReportView session={session} report={report} event={selected} onBack={() => setShowReport(false)} />
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">SNOW VAR</span>
          <span className="brand-sub">freestyle biomechanics review</span>
        </div>
        <select className="select" value={session.name} onChange={(e) => onPick(e.target.value)}>
          {index.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} · {f1(s.duration)}s · {s.events} air{s.events === 1 ? '' : 's'}
            </option>
          ))}
        </select>
        <div className="topbar-spacer" />
        <PipelineBar session={session} />
        <button className="btn btn-primary" onClick={() => setShowReport(true)} disabled={!selected}>
          Export trick breakdown
        </button>
      </header>

      <div className="layout">
        <div className="col">
          <div className="card" style={{ overflow: 'hidden' }}>
            <VideoStage
              session={session}
              videoRef={clock.ref}
              clip={api.clipUrl(session)}
              frame={frame}
              poses={poses}
              links={links}
              flags={flags}
              rpm={rpm}
              rotation={rotation}
              activeEvent={active}
            />
            <div className="transport">
              <button className="btn" onClick={clock.toggle} style={{ minWidth: 86 }}>
                {clock.playing ? '❙❙ Pause' : '▶ Play'}
              </button>
              <button className="btn btn-ghost" title="Step back one frame (←)" onClick={() => { clock.pause(); seekFrame(frame - 1) }}>
                −1
              </button>
              <button className="btn btn-ghost" title="Step forward one frame (→)" onClick={() => { clock.pause(); seekFrame(frame + 1) }}>
                +1
              </button>
              <span className="time">
                {timecode(frame / fps)} / {timecode(session.clip.duration)}
              </span>
              <select
                className="select" value={clock.rate}
                onChange={(e) => clock.setRate(Number(e.target.value))}
                title="Playback speed"
              >
                {[0.1, 0.25, 0.5, 1].map((r) => (
                  <option key={r} value={r}>{r}×</option>
                ))}
              </select>
              <div className="spacer" />
              <div className="toggles">
                {(['skeleton', 'box', 'trail'] as const).map((k) => (
                  <button
                    key={k} className="toggle" aria-pressed={flags[k]}
                    onClick={() => setFlags((f) => ({ ...f, [k]: !f[k] }))}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <span className="card-title">{selected ? `Air ${selected.id} — breakdown` : 'Clip summary'}</span>
              <span className="card-note">
                {selected ? 'metrics for the selected air' : 'select an air for its metrics'}
              </span>
            </div>
            <MetricTiles session={session} event={selected} />
          </div>

          <div className="card">
            <div className="card-head">
              <span className="card-title">Telemetry</span>
              <span className="card-note">click or drag to scrub · hatched spans are camera cuts</span>
            </div>
            <div className="card-body" style={{ paddingTop: 18 }}>
              <TelemetryTimeline
                session={session}
                frame={frame}
                onSeek={(f) => { clock.pause(); seekFrame(f) }}
                selectedId={selectedId}
                onSelect={(ev) => pickEvent(ev, false)}
              />
            </div>
          </div>
        </div>

        <div className="col">
          <div className="card">
            <div className="card-body" style={{ paddingBottom: 0 }}>
              <div className="tabs" role="tablist">
                {([['coach', 'AI coach'], ['tricks', `Airs (${session.events.length})`], ['pose', '3D pose']] as const).map(
                  ([key, label]) => (
                    <button
                      key={key} className="tab" role="tab" aria-selected={tab === key}
                      onClick={() => setTab(key)}
                    >
                      {label}
                    </button>
                  ),
                )}
              </div>
            </div>
            <div className="card-body">
              {tab === 'coach' && (
                <CoachPanel
                  session={session} report={report} event={selected}
                  loading={coachLoading} onRefresh={refreshCoach}
                />
              )}
              {tab === 'tricks' && (
                <EventList session={session} selectedId={selectedId} onSelect={(ev) => pickEvent(ev, true)} />
              )}
              {tab === 'pose' && (
                <PoseInspector3D
                  session={session} frame={frame} poses={poses} links={links} flagged={flagged}
                  onJumpToPeak={selected ? () => { clock.pause(); seekFrame(selected.peak_f) } : undefined}
                />
              )}
            </div>
            <div className="provenance">
              Pose: <code>{session.source.pose}</code> · kinematics: <code>{session.source.metrics}</code> ·{' '}
              {session.clip.tracked_frames}/{session.clip.total_frames} frames tracked
            </div>
          </div>

          <div className="card">
            <div className="card-head">
              <span className="card-title">Reading these numbers</span>
            </div>
            <div className="card-body">
              <p className="note" style={{ marginTop: 0 }}>
                {session.source.metrics === 'snow-features' ? (
                  <>
                    Rotation is <strong>{session.source.rotation_measure}</strong>, with the broadcast camera's own
                    pan estimated from the background and subtracted first. It is never integrated across a camera
                    cut — the turn that happened off-track was not measured, so it is not counted.
                  </>
                ) : (
                  <>
                    Rotation is integrated about the rider's <strong>own</strong> body axis and never across a
                    camera cut. The camera's pan has <strong>not</strong> been subtracted here, so treat the
                    absolute total as indicative — run <code>snow features</code> for the compensated number.
                  </>
                )}
              </p>
              <p className="note">
                {session.clip.slowmo_factor !== 1 ? (
                  <>
                    Rates are divided by this clip's slow-motion factor (×{session.clip.slowmo_factor}, so{' '}
                    {session.clip.real_fps} fps effective) and are real-time.
                  </>
                ) : (
                  <>
                    Rates and durations are in <strong>this clip's own</strong> timebase (
                    {session.clip.fps} fps). No slow-motion factor was given, and conformed slow motion leaves
                    no trace of its factor in the file — so if this is slow-motion footage, every rate here is
                    low by that factor and every air time high by it. Totals and angles are unaffected.
                  </>
                )}{' '}
                Metric scale is <strong>uncalibrated</strong> — angles and ratios hold, absolute metres do not.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
