'use client'

import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowLeftRight, Pause, Play, Repeat, SkipBack, SkipForward } from 'lucide-react'
import {
  advanceSlice,
  paintSlice,
  type PaintOptions,
  type PlaybackMode,
  type VolumeBundle,
} from '@/lib/volume'

export type OrganInfo = {
  id: number
  name: string
  color: string
  present: boolean
  dice?: number | null
}

const FPS_OPTIONS = [5, 10, 15, 24, 30] as const

type CaseViewerProps = {
  volume: VolumeBundle
  organs: OrganInfo[]
  hasGt: boolean
  meanDice?: number | null
  initialSlice: number
  onSliceChange: (slice: number) => void
  hotkeys: boolean
  gtBusy: string | null
}

function clampSlice(slice: number, depth: number) {
  if (depth <= 1) return 0
  return Math.max(0, Math.min(depth - 1, slice))
}

export function CaseViewer({
  volume,
  organs,
  hasGt,
  meanDice,
  initialSlice,
  onSliceChange,
  hotkeys,
  gtBusy,
}: CaseViewerProps) {
  const depth = volume.shape[0]
  const [slice, setSlice] = useState(() => clampSlice(initialSlice, depth))
  const [playing, setPlaying] = useState(false)
  const [fps, setFps] = useState<(typeof FPS_OPTIONS)[number]>(10)
  const [mode, setMode] = useState<PlaybackMode>('loop')
  const [opacity, setOpacity] = useState(45)
  const [brightness, setBrightness] = useState(0)
  const [contrast, setContrast] = useState(100)
  const [hiddenIds, setHiddenIds] = useState<number[]>([])
  const [showCt, setShowCt] = useState(true)
  const [showGt, setShowGt] = useState(true)
  const [showPred, setShowPred] = useState(true)

  const sliceRef = useRef(slice)
  const directionRef = useRef<1 | -1>(1)
  const stageRef = useRef<HTMLDivElement>(null)
  const onSliceChangeRef = useRef(onSliceChange)
  sliceRef.current = slice
  onSliceChangeRef.current = onSliceChange

  const colors: Record<number, string> = {}
  for (const organ of organs) colors[organ.id] = organ.color

  const paintOptions: PaintOptions = {
    opacity: opacity / 100,
    brightness,
    contrast,
    hiddenIds,
    colors,
  }

  function commit(next: number, pause: boolean) {
    const clamped = clampSlice(next, depth)
    sliceRef.current = clamped
    setSlice(clamped)
    onSliceChangeRef.current(clamped)
    if (pause) setPlaying(false)
  }

  useEffect(() => {
    if (!playing || depth <= 1) return
    let frame = 0
    let last = performance.now()
    let accumulator = 0
    const tick = (now: number) => {
      const elapsed = now - last
      last = now
      accumulator += elapsed
      const frameMs = 1000 / fps
      const jumps = Math.min(3, Math.floor(accumulator / frameMs))
      if (jumps > 0) {
        accumulator -= jumps * frameMs
        let next = sliceRef.current
        for (let step = 0; step < jumps; step += 1) {
          next = advanceSlice(next, depth - 1, mode, directionRef)
        }
        sliceRef.current = next
        setSlice(next)
        onSliceChangeRef.current(next)
      }
      frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [playing, fps, mode, depth])

  useEffect(() => {
    if (!hotkeys) return
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      const tag = target?.tagName
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || target?.isContentEditable) return
      if (event.code === 'Space') {
        if (tag === 'BUTTON' || tag === 'A') return
        event.preventDefault()
        if (depth <= 1) return
        setPlaying((current) => !current)
        return
      }
      if (event.key === 'ArrowRight') {
        event.preventDefault()
        commit(sliceRef.current + 1, true)
      } else if (event.key === 'ArrowLeft') {
        event.preventDefault()
        commit(sliceRef.current - 1, true)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [hotkeys, depth])

  useEffect(() => {
    const node = stageRef.current
    if (!node) return
    const onWheel = (event: WheelEvent) => {
      event.preventDefault()
      const step = Math.sign(event.deltaY)
      if (step === 0) return
      commit(sliceRef.current + step, true)
    }
    node.addEventListener('wheel', onWheel, { passive: false })
    return () => node.removeEventListener('wheel', onWheel)
  }, [depth])

  useEffect(() => {
    const onHide = () => {
      if (document.hidden) setPlaying(false)
    }
    document.addEventListener('visibilitychange', onHide)
    return () => document.removeEventListener('visibilitychange', onHide)
  }, [])

  const panels = [
    showCt ? { key: 'ct' as const, title: 'CT Input', mask: null } : null,
    hasGt && showGt && volume.gt ? { key: 'gt' as const, title: 'Ground Truth', mask: volume.gt } : null,
    showPred ? { key: 'pred' as const, title: 'Prediction', mask: volume.pred } : null,
  ].filter((panel) => panel !== null)

  const gridClass = panels.length >= 3
    ? 'lg:grid-cols-3 md:grid-cols-2'
    : panels.length === 2
      ? 'md:grid-cols-2'
      : 'grid-cols-1'

  const allHidden = organs.length > 0 && organs.every((organ) => hiddenIds.includes(organ.id))

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto p-6">
      {gtBusy && (
        <div className="flex items-center gap-2 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs text-cyan-800">
          <span className="size-3.5 animate-spin rounded-full border-2 border-cyan-600 border-t-transparent" />
          {gtBusy}
        </div>
      )}

      <div ref={stageRef} className={`grid gap-3 ${gridClass}`}>
        {panels.length === 0 && (
          <div className="flex min-h-48 items-center justify-center rounded-xl border border-dashed border-[#cbd8df] bg-[#f9fbfc] text-sm text-[#697581]">
            Turn on at least one panel.
          </div>
        )}
        {panels.map((panel) => (
          <SlicePanel
            key={panel.key}
            title={panel.title}
            ct={volume.ct}
            mask={panel.mask}
            shape={volume.shape}
            slice={slice}
            options={paintOptions}
          />
        ))}
      </div>

      <div className="rounded-xl border border-[#dbe3e8] bg-[#f9fbfc] p-3">
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            aria-label="Previous slice"
            onClick={() => commit(slice - 1, true)}
            className="flex size-8 items-center justify-center rounded-md border border-[#dbe3e8] bg-white text-[#3d4650] hover:bg-[#f1f6f8]"
          >
            <SkipBack className="size-3.5" />
          </button>
          <button
            type="button"
            aria-label={playing ? 'Pause' : 'Play'}
            aria-pressed={playing}
            disabled={depth <= 1}
            onClick={() => setPlaying((current) => !current)}
            className="flex size-9 items-center justify-center rounded-full bg-[#647184] text-white hover:bg-[#566276] disabled:opacity-40"
          >
            {playing ? <Pause className="size-4" /> : <Play className="size-4 translate-x-px" />}
          </button>
          <button
            type="button"
            aria-label="Next slice"
            onClick={() => commit(slice + 1, true)}
            className="flex size-8 items-center justify-center rounded-md border border-[#dbe3e8] bg-white text-[#3d4650] hover:bg-[#f1f6f8]"
          >
            <SkipForward className="size-3.5" />
          </button>

          <span className="w-12 shrink-0 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#7d8996]">Axial</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, depth - 1)}
            value={slice}
            aria-label="Axial slice"
            onPointerDown={() => setPlaying(false)}
            onChange={(event) => commit(Number(event.target.value), true)}
            className="h-1.5 min-w-[140px] flex-1 accent-cyan-600"
          />
          <span className="w-20 shrink-0 text-right text-xs tabular-nums text-[#5c6772]">
            {slice + 1} / {depth}
          </span>

          <label className="flex items-center gap-1.5 text-[11px] text-[#5c6772]">
            <span className="font-semibold uppercase tracking-[0.12em] text-[#7d8996]">FPS</span>
            <select
              value={fps}
              aria-label="Playback speed"
              onChange={(event) => setFps(Number(event.target.value) as (typeof FPS_OPTIONS)[number])}
              className="h-8 rounded-md border border-[#dbe3e8] bg-white px-2 text-xs text-[#29313b]"
            >
              {FPS_OPTIONS.map((value) => (
                <option key={value} value={value}>{value} FPS</option>
              ))}
            </select>
          </label>

          <button
            type="button"
            aria-label={mode === 'loop' ? 'Loop mode' : 'Ping-pong mode'}
            aria-pressed={mode === 'pingpong'}
            onClick={() => setMode((current) => (current === 'loop' ? 'pingpong' : 'loop'))}
            className="flex h-8 items-center gap-1.5 rounded-md border border-[#dbe3e8] bg-white px-2.5 text-[11px] font-semibold text-[#3d4650] hover:bg-[#f1f6f8]"
          >
            {mode === 'loop' ? <Repeat className="size-3.5" /> : <ArrowLeftRight className="size-3.5" />}
            {mode === 'loop' ? 'Loop' : 'Ping-Pong'}
          </button>
        </div>

        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2">
          <div className="flex items-center gap-1.5">
            <PanelToggle label="CT Input" pressed={showCt} onClick={() => setShowCt((value) => !value)} />
            <PanelToggle
              label="Ground Truth"
              pressed={hasGt && showGt}
              disabled={!hasGt}
              onClick={() => setShowGt((value) => !value)}
            />
            <PanelToggle label="Prediction" pressed={showPred} onClick={() => setShowPred((value) => !value)} />
          </div>
          <SliderField label="Brightness" min={-100} max={100} value={brightness} onChange={setBrightness} />
          <SliderField label="Contrast" min={50} max={200} value={contrast} display={`${contrast}%`} onChange={setContrast} />
        </div>
        <p className="mt-2 text-[10px] text-[#7d8996]">Space play/pause · ← → step · scroll on the image</p>
      </div>

      <div className="rounded-lg border border-[#dbe3e8] bg-[#f9fbfc] p-3">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#7d8996]">
              Detected Organs (BTCV)
            </div>
            {hasGt && typeof meanDice === 'number' && (
              <div className="mt-1 text-sm font-semibold text-emerald-700" title="Mean Dice over organs present in the prediction or the ground truth">
                Mean Dice {meanDice.toFixed(1)}%
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <SliderField
              label="Mask opacity"
              min={0}
              max={100}
              value={opacity}
              display={`${opacity}%`}
              onChange={setOpacity}
            />
            <button type="button" onClick={() => setHiddenIds([])} className="text-[11px] font-semibold text-cyan-700 hover:text-cyan-800">
              Show All
            </button>
            <button
              type="button"
              onClick={() => setHiddenIds(organs.map((organ) => organ.id))}
              className="text-[11px] font-semibold text-[#5c6772] hover:text-[#29313b]"
            >
              Hide All
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {organs.map((organ) => {
            const hidden = hiddenIds.includes(organ.id)
            return (
              <button
                key={organ.id}
                type="button"
                aria-pressed={!hidden}
                onClick={() => {
                  setHiddenIds((current) => (
                    current.includes(organ.id)
                      ? current.filter((id) => id !== organ.id)
                      : [...current, organ.id]
                  ))
                }}
                className={`flex items-center gap-1.5 rounded-full border px-2 py-1 text-left ${hidden ? 'border-[#e4ebf0] bg-white/60 opacity-50' : 'border-[#d5e0e6] bg-white'}`}
              >
                <span
                  className="inline-block size-3 shrink-0 rounded-sm border border-black/10"
                  style={{ backgroundColor: organ.color, opacity: organ.present ? 1 : 0.35 }}
                />
                <span className={`text-[11px] ${organ.present ? 'text-[#3d4650]' : 'text-[#9aa6b2] line-through'}`}>
                  {organ.name}
                </span>
                {hasGt && typeof organ.dice === 'number' && (
                  <span className="text-[11px] font-semibold text-[#3d4650]">({organ.dice.toFixed(1)}%)</span>
                )}
              </button>
            )
          })}
        </div>
        {allHidden && (
          <p className="mt-2 text-[10px] text-[#7d8996]">All organ overlays are hidden. CT grayscale stays visible.</p>
        )}
      </div>
    </div>
  )
}

function PanelToggle({
  label,
  pressed,
  disabled,
  onClick,
}: {
  label: string
  pressed: boolean
  disabled?: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      aria-pressed={pressed}
      disabled={disabled}
      onClick={onClick}
      className={`rounded-full px-3 py-1 text-[11px] font-semibold disabled:cursor-not-allowed disabled:opacity-40 ${pressed ? 'bg-[#647184] text-white' : 'border border-[#dbe3e8] bg-white text-[#5c6772]'}`}
    >
      {label}
    </button>
  )
}

function SliderField({
  label,
  min,
  max,
  value,
  display,
  onChange,
}: {
  label: string
  min: number
  max: number
  value: number
  display?: string
  onChange: (value: number) => void
}) {
  return (
    <label className="flex min-w-[180px] items-center gap-2 text-[11px] text-[#5c6772]">
      <span className="w-24 shrink-0">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        aria-label={label}
        onChange={(event) => onChange(Number(event.target.value))}
        className="h-1.5 w-28 accent-cyan-600"
      />
      <span className="w-10 tabular-nums">{display ?? value}</span>
    </label>
  )
}

function SlicePanel({
  title,
  ct,
  mask,
  shape,
  slice,
  options,
}: {
  title: string
  ct: Uint8Array
  mask: Uint8Array | null
  shape: VolumeBundle['shape']
  slice: number
  options: PaintOptions
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const frameRef = useRef<ImageData | null>(null)
  const [, height, width] = shape

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    frameRef.current = paintSlice(canvas, ct, mask, shape, slice, options, frameRef.current)
  }, [ct, mask, shape, slice, options])

  return (
    <figure className="min-w-0">
      <figcaption className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-[#7d8996]">
        {title}
      </figcaption>
      <div className="mx-auto w-full" style={{ maxWidth: 'min(100%, 74vh)' }}>
        <div
          className="relative w-full overflow-hidden rounded-xl border border-[#dbe3e8] bg-[#10161a]"
          style={{ aspectRatio: `${width} / ${height}` }}
        >
          <canvas ref={canvasRef} className="absolute inset-0 block h-full w-full object-contain" role="img" aria-label={title} />
        </div>
      </div>
    </figure>
  )
}
