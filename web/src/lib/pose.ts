import type { PoseFrame, Session } from '../types'

/** The Nova-77 skeleton carries five finger chains per hand. They are real data
 * but unreadable at any sane on-screen size, so the overlay and the inspector
 * draw the body chain only. */
export const CORE_JOINTS: number[] = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, // pelvis -> spine -> neck -> head
  11, 12, 13, 14, // one arm: clavicle, shoulder, elbow, wrist
  39, 40, 41, 42, // the other arm
  67, 68, 69, 70, 71, // one leg: hip, knee, ankle, foot, toe
  72, 73, 74, 75, 76, // the other leg
]

export type Group = 'torso' | 'arms' | 'legs'

export function groupOf(joint: number): Group {
  if (joint >= 67) return 'legs'
  if ((joint >= 11 && joint <= 14) || (joint >= 39 && joint <= 42)) return 'arms'
  return 'torso'
}

export const GROUP_COLOR: Record<Group, string> = {
  torso: '#f0efec',
  arms: '#3987e5',
  legs: '#199e70',
}

export function coreLinks(session: Session): [number, number][] {
  const core = new Set(CORE_JOINTS)
  return session.skeleton.links.filter(([a, b]) => core.has(a) && core.has(b))
}

export function frameIndex(session: Session): Map<number, PoseFrame> {
  return new Map(session.frames.map((fr) => [fr.f, fr]))
}

/** Index into the telemetry arrays for a frame id, or -1 when untracked. */
export function telemetryIndex(session: Session): Map<number, number> {
  const m = new Map<number, number>()
  session.telemetry.f.forEach((f, i) => m.set(f, i))
  return m
}

export interface Vec3 {
  x: number
  y: number
  z: number
}

/** Millimetres in the NIM's camera frame -> metres in a viewer frame where
 * +Y is up. The NIM returns Y pointing down (OpenCV convention). */
export function toViewer(k: [number, number, number]): Vec3 {
  return { x: k[0] / 1000, y: -k[1] / 1000, z: k[2] / 1000 }
}

export const sub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z })
export const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z })
export const scaleV = (a: Vec3, k: number): Vec3 => ({ x: a.x * k, y: a.y * k, z: a.z * k })
export const norm = (a: Vec3): number => Math.hypot(a.x, a.y, a.z)

export function unit(a: Vec3): Vec3 {
  const n = norm(a)
  return n > 1e-9 ? scaleV(a, 1 / n) : a
}

export const mid = (a: Vec3, b: Vec3): Vec3 => scaleV(add(a, b), 0.5)

/** Orthographic projection after a yaw then a pitch — enough for an orbitable
 * inspector, and it needs no WebGL context to stay reliable in a demo. */
export function project(p: Vec3, azimuth: number, elevation: number): { x: number; y: number; depth: number } {
  const ca = Math.cos(azimuth)
  const sa = Math.sin(azimuth)
  const x1 = p.x * ca + p.z * sa
  const z1 = -p.x * sa + p.z * ca
  const ce = Math.cos(elevation)
  const se = Math.sin(elevation)
  const y2 = p.y * ce - z1 * se
  const z2 = p.y * se + z1 * ce
  return { x: x1, y: y2, depth: z2 }
}
