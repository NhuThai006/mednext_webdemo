export type VolumeKind = 'ct' | 'pred' | 'gt'

export type VolumeShape = [number, number, number]

export type VolumeBundle = {
  shape: VolumeShape
  ct: Uint8Array
  pred: Uint8Array
  gt: Uint8Array | null
}

export type PlaybackMode = 'loop' | 'pingpong'

export const BTCV_HEX = [
  '#000000',
  '#4CAF50',
  '#7EB8F7',
  '#5B8CFF',
  '#FF8C42',
  '#FF6B9D',
  '#E8D44D',
  '#1E90FF',
  '#FF4444',
  '#FF9F43',
  '#A29BFE',
  '#B57BFF',
  '#00CEC9',
  '#55EFC4',
]

const bundles = new Map<string, VolumeBundle>()

const littleEndian = (() => {
  const probe = new Uint32Array([0xff])
  return new Uint8Array(probe.buffer)[0] === 0xff
})()

export function peekVolumes(jobId: string) {
  return bundles.get(jobId) ?? null
}

export function dropVolumes(jobId: string) {
  bundles.delete(jobId)
}

export function advanceSlice(
  slice: number,
  max: number,
  mode: PlaybackMode,
  direction: { current: 1 | -1 },
) {
  if (max <= 0) return 0
  if (mode === 'loop') return slice >= max ? 0 : slice + 1
  let dir = direction.current
  if (dir > 0 && slice >= max) dir = -1
  else if (dir < 0 && slice <= 0) dir = 1
  let next = slice + dir
  if (next > max) next = max
  if (next < 0) next = 0
  direction.current = dir
  return next
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace('#', '')
  if (value.length !== 6) return [255, 255, 255]
  return [
    Number.parseInt(value.slice(0, 2), 16),
    Number.parseInt(value.slice(2, 4), 16),
    Number.parseInt(value.slice(4, 6), 16),
  ]
}

async function inflateIfNeeded(bytes: Uint8Array, expected: number) {
  if (bytes.length === expected) return bytes
  const gzip = bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b
  if (!gzip || typeof DecompressionStream === 'undefined') return bytes
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))
  return new Uint8Array(await new Response(stream).arrayBuffer())
}

async function fetchVolume(backend: string, jobId: string, kind: VolumeKind) {
  const response = await fetch(`${backend}/api/volume/${jobId}/${kind}`)
  if (!response.ok) {
    let detail = `Could not download the ${kind} volume.`
    try {
      const body = await response.json() as { detail?: unknown }
      if (typeof body.detail === 'string' && body.detail.trim()) detail = body.detail
    } catch {
      // Response may be empty or binary.
    }
    throw new Error(detail)
  }

  const shapeHeader = response.headers.get('X-Volume-Shape') ?? response.headers.get('x-volume-shape')
  if (!shapeHeader) throw new Error('Volume response is missing its shape header.')
  const parts = shapeHeader.split(',').map((part) => Number(part))
  if (parts.length !== 3 || parts.some((part) => !Number.isInteger(part) || part <= 0)) {
    throw new Error('Volume shape header is invalid.')
  }
  const shape = parts as VolumeShape
  const expected = shape[0] * shape[1] * shape[2]
  const raw = new Uint8Array(await response.arrayBuffer())
  const data = await inflateIfNeeded(raw, expected)
  if (data.length !== expected) {
    throw new Error(`Volume size mismatch: received ${data.length} bytes, expected ${expected}.`)
  }
  return { data, shape }
}

export async function loadVolumes(backend: string, jobId: string, hasGt: boolean, refreshGt = false) {
  const existing = bundles.get(jobId)
  let shape: VolumeShape
  let ct: Uint8Array
  let pred: Uint8Array

  if (existing && !refreshGt) {
    shape = existing.shape
    ct = existing.ct
    pred = existing.pred
  } else if (existing && refreshGt) {
    shape = existing.shape
    ct = existing.ct
    pred = existing.pred
  } else {
    const [ctVol, predVol] = await Promise.all([
      fetchVolume(backend, jobId, 'ct'),
      fetchVolume(backend, jobId, 'pred'),
    ])
    if (ctVol.shape[0] !== predVol.shape[0] || ctVol.shape[1] !== predVol.shape[1] || ctVol.shape[2] !== predVol.shape[2]) {
      throw new Error('CT and prediction volumes have different shapes.')
    }
    shape = ctVol.shape
    ct = ctVol.data
    pred = predVol.data
  }

  let gt: Uint8Array | null = refreshGt ? null : existing?.gt ?? null
  if (hasGt && !gt) {
    const gtVol = await fetchVolume(backend, jobId, 'gt')
    if (gtVol.shape[0] !== shape[0] || gtVol.shape[1] !== shape[1] || gtVol.shape[2] !== shape[2]) {
      throw new Error('Ground truth volume shape does not match the CT volume.')
    }
    gt = gtVol.data
  }
  if (!hasGt) gt = null

  const bundle: VolumeBundle = { shape, ct, pred, gt }
  bundles.set(jobId, bundle)
  return bundle
}

export type PaintOptions = {
  opacity: number
  brightness: number
  contrast: number
  hiddenIds: readonly number[]
  colors: Record<number, string>
}

function buildBlendLut(options: PaintOptions) {
  const gray = new Uint8Array(256)
  const scale = options.contrast / 100
  for (let value = 0; value < 256; value += 1) {
    let adjusted = (value - 128) * scale + 128 + options.brightness
    if (adjusted < 0) adjusted = 0
    else if (adjusted > 255) adjusted = 255
    gray[value] = adjusted
  }

  const hidden = new Uint8Array(14)
  for (const id of options.hiddenIds) {
    if (id > 0 && id < 14) hidden[id] = 1
  }

  const palette: Array<[number, number, number]> = BTCV_HEX.map((hex) => hexToRgb(hex))
  for (const [id, hex] of Object.entries(options.colors)) {
    const label = Number(id)
    if (label > 0 && label < 14) palette[label] = hexToRgb(hex)
  }

  const lut = new Uint32Array(14 * 256)
  const opacity = Math.min(1, Math.max(0, options.opacity))
  for (let raw = 0; raw < 256; raw += 1) {
    const base = gray[raw]
    for (let label = 0; label < 14; label += 1) {
      let red = base
      let green = base
      let blue = base
      if (label > 0 && !hidden[label] && opacity > 0) {
        const [cr, cg, cb] = palette[label]
        const keep = 1 - opacity
        red = Math.round(base * keep + cr * opacity)
        green = Math.round(base * keep + cg * opacity)
        blue = Math.round(base * keep + cb * opacity)
      }
      const packed = littleEndian
        ? (red | (green << 8) | (blue << 16) | (255 << 24)) >>> 0
        : ((red << 24) | (green << 16) | (blue << 8) | 255) >>> 0
      lut[label * 256 + raw] = packed
    }
  }
  return lut
}

export function paintSlice(
  canvas: HTMLCanvasElement,
  ct: Uint8Array,
  mask: Uint8Array | null,
  shape: VolumeShape,
  slice: number,
  options: PaintOptions,
  image: ImageData | null,
) {
  const [depth, height, width] = shape
  const z = Math.max(0, Math.min(depth - 1, slice))
  const ctx = canvas.getContext('2d', { alpha: false })
  if (!ctx) return image

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
    image = null
  }
  const frame = image && image.width === width && image.height === height
    ? image
    : ctx.createImageData(width, height)
  const lut = buildBlendLut(options)
  const plane = height * width
  const offset = z * plane
  const pixels = frame.data

  if (littleEndian) {
    const view = new Uint32Array(pixels.buffer, pixels.byteOffset, plane)
    if (mask) {
      for (let index = 0; index < plane; index += 1) {
        let label = mask[offset + index]
        if (label > 13) label = 0
        view[index] = lut[(label << 8) + ct[offset + index]]
      }
    } else {
      for (let index = 0; index < plane; index += 1) {
        view[index] = lut[ct[offset + index]]
      }
    }
  } else {
    for (let index = 0; index < plane; index += 1) {
      let label = mask ? mask[offset + index] : 0
      if (label > 13) label = 0
      const packed = lut[(label << 8) + ct[offset + index]]
      const pixel = index * 4
      pixels[pixel] = (packed >>> 24) & 255
      pixels[pixel + 1] = (packed >>> 16) & 255
      pixels[pixel + 2] = (packed >>> 8) & 255
      pixels[pixel + 3] = 255
    }
  }

  ctx.putImageData(frame, 0, 0)
  return frame
}
