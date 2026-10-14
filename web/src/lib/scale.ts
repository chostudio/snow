/** Minimal scale + path helpers. No chart library: every mark here is one we
 * place deliberately, which is also what keeps the bundle small. */

export interface Scale {
  (v: number): number
  invert(px: number): number
  domain: [number, number]
  range: [number, number]
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain
  const [r0, r1] = range
  const span = d1 - d0 || 1
  const s = ((v: number) => r0 + ((v - d0) / span) * (r1 - r0)) as Scale
  s.invert = (px: number) => d0 + ((px - r0) / (r1 - r0 || 1)) * span
  s.domain = domain
  s.range = range
  return s
}

/** A domain padded to round numbers and always including zero for signed data. */
export function niceDomain(values: number[], includeZero: boolean): [number, number] {
  let lo = Infinity
  let hi = -Infinity
  for (const v of values) {
    if (!Number.isFinite(v)) continue
    if (v < lo) lo = v
    if (v > hi) hi = v
  }
  if (!Number.isFinite(lo)) return [0, 1]
  if (includeZero) {
    lo = Math.min(0, lo)
    hi = Math.max(0, hi)
  }
  if (lo === hi) {
    lo -= 1
    hi += 1
  }
  const pad = (hi - lo) * 0.08
  const step = Math.pow(10, Math.floor(Math.log10(hi - lo || 1))) / 2
  return [Math.floor((lo - pad) / step) * step, Math.ceil((hi + pad) / step) * step]
}

export function ticksEvery(domain: [number, number], step: number): number[] {
  const out: number[] = []
  for (let v = Math.ceil(domain[0] / step) * step; v <= domain[1] + 1e-9; v += step) {
    out.push(Math.round(v * 1e6) / 1e6)
  }
  return out
}

/** Build one path per contiguous run of samples.
 *
 * Splitting matters: a camera cut leaves the rider untracked, and a line drawn
 * straight across that hole would read as measured data. */
export function segmentedPath(
  xs: number[],
  ys: number[],
  isContiguous: (i: number) => boolean,
  sx: Scale,
  sy: Scale,
): string {
  let d = ''
  for (let i = 0; i < xs.length; i++) {
    const cmd = i === 0 || !isContiguous(i) ? 'M' : 'L'
    d += `${cmd}${sx(xs[i]).toFixed(1)},${sy(ys[i]).toFixed(1)}`
  }
  return d
}
