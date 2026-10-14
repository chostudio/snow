import { useEffect, useRef, useState } from 'react'
import type { PoseFrame, Session } from '../types'
import {
  add, CORE_JOINTS, GROUP_COLOR, groupOf, mid, norm, project, scaleV, sub, toViewer, unit,
  type Vec3,
} from '../lib/pose'

import { f1 } from '../lib/format'

interface Props {
  session: Session
  frame: number
  poses: Map<number, PoseFrame>
  links: [number, number][]
  /** Joints to call out, e.g. knees that are extended at touchdown. */
  flagged: Set<number>
  onJumpToPeak?: () => void
}

const VIEWS: { label: string; az: number; el: number }[] = [
  { label: 'Front', az: 0, el: 0 },
  { label: 'Side', az: Math.PI / 2, el: 0 },
  { label: 'Top', az: 0, el: -1.45 },
]

const W = 364
const H = 288

export default function PoseInspector3D({ session, frame, poses, links, flagged, onJumpToPeak }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [view, setView] = useState({ az: 0.35, el: 0.12 })
  const drag = useRef<{ x: number; y: number } | null>(null)
  const pose = poses.get(frame) ?? null

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const dpr = window.devicePixelRatio || 1
    canvas.width = W * dpr
    canvas.height = H * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, W, H)

    if (!pose) {
      ctx.fillStyle = '#898781'
      ctx.font = '12px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('No rider tracked at this frame', W / 2, H / 2)
      return
    }

    const pts: Vec3[] = pose.k3.map(toViewer)
    const J = session.skeleton.joints
    const hipMid = mid(pts[J.hip_l], pts[J.hip_r])
    const shoMid = mid(pts[J.shoulder_l], pts[J.shoulder_r])
    const bodyUp = unit(sub(shoMid, hipMid))

    // Reference verticals: the camera's own up, and the rider's body axis. The
    // angle between them is the corking measure the tiles report.
    const along = (base: Vec3, dir: Vec3, d: number): Vec3 => add(base, scaleV(dir, d))
    const camUp = { x: 0, y: 1, z: 0 }
    const vTop = along(hipMid, camUp, 0.6)
    const vBot = along(hipMid, camUp, -0.5)
    const aTop = along(hipMid, bodyUp, 0.6)

    // Fit the projected bounding box, not a radius about the pelvis: a corked
    // rider is wildly asymmetric about their own hips, and centring on a joint
    // parks half the body off-frame. Two passes — measure, then place.
    const fit = [...CORE_JOINTS.map((j) => pts[j]), vTop, vBot, aTop]
    const flat = fit.map((p) => project(p, view.az, view.el))
    const xs = flat.map((q) => q.x)
    const ys = flat.map((q) => q.y)
    const bx0 = Math.min(...xs)
    const bx1 = Math.max(...xs)
    const by0 = Math.min(...ys)
    const by1 = Math.max(...ys)
    const scale = Math.min((W - 40) / Math.max(0.2, bx1 - bx0), (H - 40) / Math.max(0.2, by1 - by0))
    const bcx = (bx0 + bx1) / 2
    const bcy = (by0 + by1) / 2

    const to2 = (p: Vec3) => {
      const q = project(p, view.az, view.el)
      return { x: W / 2 + (q.x - bcx) * scale, y: H / 2 - (q.y - bcy) * scale, depth: q.depth }
    }

    ctx.strokeStyle = '#4a4a46'
    ctx.lineWidth = 1
    ctx.setLineDash([4, 5])
    ctx.beginPath()
    ctx.moveTo(to2(vTop).x, to2(vTop).y)
    ctx.lineTo(to2(vBot).x, to2(vBot).y)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.strokeStyle = '#d95926'
    ctx.lineWidth = 1.75
    ctx.beginPath()
    ctx.moveTo(to2(hipMid).x, to2(hipMid).y)
    ctx.lineTo(to2(aTop).x, to2(aTop).y)
    ctx.stroke()

    const tilt = (Math.acos(Math.max(-1, Math.min(1, bodyUp.y / (norm(bodyUp) || 1)))) * 180) / Math.PI
    ctx.fillStyle = '#d95926'
    ctx.font = '600 11px system-ui, sans-serif'
    ctx.textAlign = 'left'
    ctx.fillText(`axis ${f1(tilt)}° off vertical`, 10, 18)

    // Painter's algorithm: far bones first, so the near side reads as near.
    const ordered = links
      .map(([a, b]) => ({ a, b, d: (to2(pts[a]).depth + to2(pts[b]).depth) / 2 }))
      .sort((p, q) => p.d - q.d)

    for (const { a, b, d } of ordered) {
      const near = Math.max(0.25, Math.min(1, 0.6 + d * 0.5))
      const pa = to2(pts[a])
      const pb = to2(pts[b])
      ctx.globalAlpha = near
      ctx.strokeStyle = GROUP_COLOR[groupOf(a)]
      ctx.lineWidth = 2 + near * 1.6
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(pa.x, pa.y)
      ctx.lineTo(pb.x, pb.y)
      ctx.stroke()
    }
    ctx.globalAlpha = 1

    for (const [j] of links) {
      const p = to2(pts[j])
      ctx.fillStyle = flagged.has(j) ? '#d03b3b' : GROUP_COLOR[groupOf(j)]
      ctx.beginPath()
      ctx.arc(p.x, p.y, flagged.has(j) ? 5 : 2.6, 0, Math.PI * 2)
      ctx.fill()
      if (flagged.has(j)) {
        ctx.strokeStyle = '#d03b3b'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.arc(p.x, p.y, 9, 0, Math.PI * 2)
        ctx.stroke()
      }
    }
  }, [pose, view, links, session, flagged])

  return (
    <div>
      <div className="inspector">
        <canvas
          ref={canvasRef} style={{ width: W, height: H, maxWidth: '100%' }}
          onPointerDown={(e) => {
            drag.current = { x: e.clientX, y: e.clientY }
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerMove={(e) => {
            if (!drag.current) return
            const dx = e.clientX - drag.current.x
            const dy = e.clientY - drag.current.y
            drag.current = { x: e.clientX, y: e.clientY }
            setView((v) => ({
              az: v.az + dx * 0.01,
              el: Math.max(-1.5, Math.min(1.5, v.el + dy * 0.01)),
            }))
          }}
          onPointerUp={() => {
            drag.current = null
          }}
        />
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 10, flexWrap: 'wrap' }}>
        <div className="views">
          {VIEWS.map((v) => (
            <button key={v.label} className="toggle" onClick={() => setView({ az: v.az, el: v.el })}>
              {v.label}
            </button>
          ))}
        </div>
        {onJumpToPeak && (
          <button className="toggle" onClick={onJumpToPeak} style={{ marginLeft: 'auto' }}>
            Jump to peak rate
          </button>
        )}
      </div>
      <div className="legend" style={{ marginTop: 10 }}>
        <span><i style={{ background: GROUP_COLOR.torso }} />torso</span>
        <span><i style={{ background: GROUP_COLOR.arms }} />arms</span>
        <span><i style={{ background: GROUP_COLOR.legs }} />legs</span>
        <span><i style={{ background: '#d95926' }} />body axis</span>
        <span><i style={{ background: '#d03b3b', borderRadius: '50%' }} />flagged joint</span>
      </div>
      <p className="note" style={{ marginTop: 8 }}>
        Drag to orbit. Fingers are omitted — the NIM returns all{' '}
        {session.skeleton.num_joints} Nova joints, but only the body chain is legible here.
      </p>
    </div>
  )
}
