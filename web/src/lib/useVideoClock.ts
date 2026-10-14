import { useCallback, useEffect, useRef, useState } from 'react'

/** The video element is the clock for everything on screen.
 *
 * Telemetry, the skeleton overlay and the playhead all read the same frame
 * number derived from `currentTime`, so nothing can drift out of sync with the
 * footage — which is the whole point of a review tool. */
export function useVideoClock(fps: number, totalFrames: number) {
  const ref = useRef<HTMLVideoElement | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [rate, setRate] = useState(1)

  useEffect(() => {
    let raf = 0
    const tick = () => {
      const v = ref.current
      // floor, not round: seeks land mid-frame, and rounding a mid-frame time
      // up would make a single step advance two frames.
      if (v) setFrame(Math.min(totalFrames - 1, Math.floor(v.currentTime * fps)))
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [fps, totalFrames])

  useEffect(() => {
    const v = ref.current
    if (!v) return
    const onPlay = () => setPlaying(true)
    const onPause = () => setPlaying(false)
    v.addEventListener('play', onPlay)
    v.addEventListener('pause', onPause)
    return () => {
      v.removeEventListener('play', onPlay)
      v.removeEventListener('pause', onPause)
    }
  })

  useEffect(() => {
    if (ref.current) ref.current.playbackRate = rate
  }, [rate])

  const seekFrame = useCallback(
    (f: number) => {
      const v = ref.current
      if (!v) return
      const clamped = Math.max(0, Math.min(totalFrames - 1, Math.round(f)))
      // Mid-frame keeps the seek from landing on a boundary and snapping back.
      v.currentTime = (clamped + 0.5) / fps
    },
    [fps, totalFrames],
  )

  const toggle = useCallback(() => {
    const v = ref.current
    if (!v) return
    if (v.paused) void v.play()
    else v.pause()
  }, [])

  const pause = useCallback(() => ref.current?.pause(), [])

  return { ref, frame, playing, rate, setRate, seekFrame, toggle, pause }
}
