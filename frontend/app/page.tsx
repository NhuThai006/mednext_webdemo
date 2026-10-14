'use client'

import { useEffect, useRef, useState } from 'react'
import { CheckCircle2, FileUp, FolderOpen, LoaderCircle, Upload, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CaseViewer, type OrganInfo } from '@/components/case-viewer'
import { dropVolumes, loadVolumes, peekVolumes } from '@/lib/volume'

const acceptedExtensions = ['.nii', '.nii.gz']

// Đổi thành http://<IP_SERVER>:8000 nếu không dùng SSH tunnel.
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'

type CaseStatus = 'uploading' | 'processing' | 'downloading' | 'done' | 'error'

type CaseFile = {
  id: string
  file: File
  gtFile?: File
  name: string
  gtName?: string
  size: string
  status: CaseStatus
  jobId?: string
  downloadUrl?: string
  totalSlices?: number
  organs?: OrganInfo[]
  processingTime?: number
  cached?: boolean
  hasGt?: boolean
  meanDice?: number | null
  errorMessage?: string
  gtBusy?: string | null
}

type SegmentResponse = {
  detail?: unknown
  job_id?: string
  total_slices?: number
  organs_detected?: OrganInfo[]
  processing_time_seconds?: number
  cached?: boolean
  has_gt?: boolean
  mean_dice?: number | null
}

function isNiftiFile(name: string) {
  return acceptedExtensions.some((extension) => name.toLowerCase().endsWith(extension))
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function statusLabel(status: CaseStatus) {
  if (status === 'done') return 'Ready'
  if (status === 'error') return 'Error'
  if (status === 'uploading') return 'Uploading'
  if (status === 'downloading') return 'Loading volume'
  return 'Inference'
}

function phaseMessage(status: CaseStatus) {
  if (status === 'uploading') return 'Uploading...'
  if (status === 'downloading') return 'Downloading 3D Volume to Browser Memory...'
  if (status === 'processing') return 'Running MedNeXt Inference (or Loading from Cache)...'
  if (status === 'done') return 'Ready'
  return 'Error'
}

function statusClass(status: CaseStatus) {
  if (status === 'done') return 'text-emerald-600'
  if (status === 'error') return 'text-red-500'
  return 'text-cyan-600'
}

function detailMessage(detail: unknown, fallback: string) {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    const text = detail.map((item) => {
      if (typeof item === 'string') return item
      if (item && typeof item === 'object' && 'msg' in item) return String((item as { msg: unknown }).msg)
      return ''
    }).filter(Boolean).join(' ')
    if (text) return text
  }
  return fallback
}

function postForm(
  url: string,
  form: FormData,
  onPhase?: (phase: 'uploading' | 'processing') => void,
) {
  return new Promise<{ ok: boolean, status: number, json: SegmentResponse }>((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', url)
    xhr.responseType = 'text'
    onPhase?.('uploading')
    xhr.upload.onload = () => onPhase?.('processing')
    xhr.onerror = () => reject(new Error('Failed to fetch'))
    xhr.onload = () => {
      let json: SegmentResponse = {}
      try {
        json = JSON.parse(xhr.responseText) as SegmentResponse
      } catch {
        json = {}
      }
      resolve({ ok: xhr.status >= 200 && xhr.status < 300, status: xhr.status, json })
    }
    xhr.send(form)
  })
}

export default function Page() {
  const [cases, setCases] = useState<CaseFile[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [notice, setNotice] = useState('')
  const [uploadOpen, setUploadOpen] = useState(false)
  const [pendingCt, setPendingCt] = useState<File | null>(null)
  const [volumeVersion, setVolumeVersion] = useState(0)
  const sliceMemory = useRef<Record<string, number>>({})
  const ctPickerRef = useRef<HTMLInputElement>(null)
  const addGtRef = useRef<HTMLInputElement>(null)

  const selectedCase = cases.find((item) => item.id === selectedId) ?? cases[0] ?? null
  const hasCases = cases.length > 0
  const bundle = selectedCase?.jobId ? peekVolumes(selectedCase.jobId) : null
  void volumeVersion

  useEffect(() => {
    if (!uploadOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setUploadOpen(false)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [uploadOpen])

  useEffect(() => {
    if (!selectedCase || selectedCase.status !== 'done' || !selectedCase.jobId) return
    if (peekVolumes(selectedCase.jobId)) return
    const caseId = selectedCase.id
    const jobId = selectedCase.jobId
    const hasGt = Boolean(selectedCase.hasGt)
    let cancelled = false
    void (async () => {
      try {
        await loadVolumes(BACKEND, jobId, hasGt)
        if (!cancelled) setVolumeVersion((value) => value + 1)
      } catch (err: unknown) {
        if (cancelled) return
        const raw = err instanceof Error ? err.message : 'Could not download the volume.'
        setCases((prev) => prev.map((item) => (
          item.id === caseId ? { ...item, status: 'error', errorMessage: raw } : item
        )))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedCase?.id, selectedCase?.status, selectedCase?.jobId, selectedCase?.hasGt])

  function showNotice(message: string) {
    setNotice(message)
    window.setTimeout(() => setNotice(''), 4200)
  }

  function updateCase(caseId: string, patch: Partial<CaseFile>) {
    setCases((prev) => prev.map((item) => (item.id === caseId ? { ...item, ...patch } : item)))
  }

  async function uploadAndSegment(file: File, caseId: string, gtFile?: File) {
    const formData = new FormData()
    formData.append('file', file)
    if (gtFile) formData.append('gt_file', gtFile)

    try {
      const response = await postForm(`${BACKEND}/api/segment`, formData, (phase) => {
        updateCase(caseId, { status: phase, errorMessage: undefined })
      })
      const data = response.json
      const jobId = data.job_id
      const totalSlices = data.total_slices
      if (!response.ok || !jobId || totalSlices == null) {
        throw new Error(detailMessage(data.detail, 'Segmentation failed'))
      }

      updateCase(caseId, { status: 'downloading', jobId, totalSlices })
      await loadVolumes(BACKEND, jobId, Boolean(data.has_gt))
      setVolumeVersion((value) => value + 1)
      updateCase(caseId, {
        status: 'done',
        jobId,
        downloadUrl: `${BACKEND}/api/result/${jobId}`,
        totalSlices,
        organs: data.organs_detected,
        processingTime: data.processing_time_seconds,
        cached: Boolean(data.cached),
        hasGt: Boolean(data.has_gt),
        meanDice: data.mean_dice,
        gtName: gtFile?.name,
        errorMessage: undefined,
      })
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : 'Unknown error'
      const message = raw === 'Failed to fetch'
        ? `Cannot reach the segmentation server at ${BACKEND}.`
        : raw
      updateCase(caseId, { status: 'error', errorMessage: message })
    }
  }

  function addFiles(fileList: FileList | File[], gtFile?: File) {
    const files = Array.from(fileList)
    const invalid = files.find((file) => !isNiftiFile(file.name))
    if (invalid) showNotice('Only .nii and .nii.gz files are supported.')
    if (gtFile && !isNiftiFile(gtFile.name)) {
      showNotice('Ground truth must be a .nii or .nii.gz file.')
      return
    }
    const validFiles = files.filter((file) => isNiftiFile(file.name))
    if (!validFiles.length) return

    const nextCases: CaseFile[] = validFiles.map((file, index) => ({
      id: `${file.name}-${file.lastModified}-${Math.random()}`,
      file,
      gtFile: index === 0 ? gtFile : undefined,
      gtName: index === 0 ? gtFile?.name : undefined,
      name: file.name,
      size: formatFileSize(file.size),
      status: 'uploading',
    }))

    setCases((current) => [...current, ...nextCases])
    setSelectedId(nextCases[0].id)
    nextCases.forEach((item) => {
      void uploadAndSegment(item.file, item.id, item.gtFile)
    })
  }

  function removeCase(id: string) {
    const target = cases.find((entry) => entry.id === id)
    const jobStillUsed = cases.some((entry) => entry.id !== id && entry.jobId && entry.jobId === target?.jobId)
    if (target?.jobId && !jobStillUsed) dropVolumes(target.jobId)
    setCases((current) => current.filter((entry) => entry.id !== id))
    setSelectedId((current) => (current === id ? null : current))
  }

  function closeUpload() {
    setUploadOpen(false)
    setPendingCt(null)
    if (ctPickerRef.current) ctPickerRef.current.value = ''
  }

  function submitUpload() {
    if (!pendingCt) {
      showNotice('Choose a CT volume before starting.')
      return
    }
    addFiles([pendingCt])
    closeUpload()
  }

  async function addGroundTruth(file: File) {
    if (!selectedCase?.jobId) return
    if (!isNiftiFile(file.name)) {
      showNotice('Ground truth must be a .nii or .nii.gz file.')
      return
    }
    const caseId = selectedCase.id
    const jobId = selectedCase.jobId
    updateCase(caseId, { gtBusy: 'Uploading ground truth...' })
    const formData = new FormData()
    formData.append('gt_file', file)
    try {
      const response = await postForm(`${BACKEND}/api/upload_gt/${jobId}`, formData, (phase) => {
        updateCase(caseId, {
          gtBusy: phase === 'uploading' ? 'Uploading ground truth...' : 'Computing Dice...',
        })
      })
      if (!response.ok || !response.json.job_id) {
        throw new Error(detailMessage(response.json.detail, 'Ground truth upload failed'))
      }
      updateCase(caseId, { gtBusy: 'Downloading 3D Volume to Browser Memory...' })
      await loadVolumes(BACKEND, jobId, true, true)
      setVolumeVersion((value) => value + 1)
      updateCase(caseId, {
        gtBusy: null,
        hasGt: Boolean(response.json.has_gt),
        meanDice: response.json.mean_dice,
        organs: response.json.organs_detected ?? selectedCase.organs,
        gtName: file.name,
        gtFile: file,
      })
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : 'Ground truth upload failed'
      updateCase(caseId, { gtBusy: null })
      showNotice(raw === 'Failed to fetch' ? `Cannot reach the segmentation server at ${BACKEND}.` : raw)
    }
  }

  const showPhase = selectedCase && (selectedCase.status === 'uploading' || selectedCase.status === 'processing' || selectedCase.status === 'downloading')
  const showRecover = selectedCase?.status === 'done' && selectedCase.jobId && !bundle

  return (
    <main className="min-h-screen bg-[#f4f7f9] text-[#29313b] selection:bg-[#647184]/20">
      <div className="flex min-h-screen">
        <aside className="flex w-[272px] shrink-0 flex-col border-r border-[#dbe3e8] bg-[#edf2f5]">
          <header className="flex h-[72px] items-center gap-3 border-b border-[#dbe3e8] px-6">
            <div className="flex size-9 items-center justify-center overflow-hidden rounded-lg bg-cyan-400/10">
              <img src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-QsvcipHMR89sTeyUBgUem1DsBNv7VS.png" alt="MedNext logo" className="size-full object-cover" />
            </div>
            <div>
              <div className="text-[15px] font-semibold tracking-[-0.01em]">MedNext</div>
              <div className="mt-0.5 text-[10px] font-medium uppercase tracking-[0.18em] text-[#6c7784]">CT Organ Segmentation</div>
            </div>
          </header>

          <div className="flex items-center justify-between border-b border-[#dbe3e8] px-5 py-5">
            <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-[#7d8996]">Cases</span>
            <Button onClick={() => setUploadOpen(true)} size="sm" className="h-8 gap-2 rounded-md bg-[#647184] px-3 text-[12px] font-semibold text-white hover:bg-[#566276]">
              <Upload data-icon="inline-start" /> Upload
            </Button>
          </div>

          <div className="flex flex-1 flex-col items-center overflow-auto px-5 pt-5 text-center">
            {!hasCases ? (
              <div className="pt-16">
                <FolderOpen className="mx-auto mb-3 size-7 text-[#46515e]" aria-hidden="true" />
                <p className="text-[13px] text-[#6f7b88]">No cases loaded</p>
              </div>
            ) : (
              <div className="flex w-full flex-col gap-2 text-left">
                {cases.map((item) => {
                  const active = selectedCase?.id === item.id
                  return (
                    <div key={item.id} className={`rounded-lg border bg-[#f9fbfc] p-3 ${active ? 'border-cyan-400' : 'border-[#dfe7ec]'}`}>
                      <div className="flex items-start gap-2">
                        <button type="button" onClick={() => setSelectedId(item.id)} className="flex min-w-0 flex-1 items-start gap-2 text-left">
                          {item.status === 'processing' || item.status === 'uploading' || item.status === 'downloading'
                            ? <LoaderCircle className="mt-0.5 size-4 shrink-0 animate-spin text-cyan-500" />
                            : <FileUp className="mt-0.5 size-4 shrink-0 text-cyan-600" />}
                          <span className="min-w-0 flex-1 truncate text-[12px]">{item.name}</span>
                        </button>
                        <button type="button" onClick={() => removeCase(item.id)} aria-label={`Remove ${item.name}`} className="text-[#66717d] hover:text-[#29313b]">
                          <X className="size-3.5" />
                        </button>
                      </div>
                      <div className="mt-2 flex items-center gap-2 pl-6 text-[10px] text-[#6f7b88]">
                        <span>{item.size}</span>
                        <span className="text-[#39424c]">·</span>
                        <span className={statusClass(item.status)}>{statusLabel(item.status)}</span>
                        {item.cached && item.status === 'done' && <span className="text-cyan-600">⚡</span>}
                      </div>
                      {item.gtName && <div className="mt-1 truncate pl-6 text-[10px] text-[#6f7b88]">GT: {item.gtName}</div>}
                    </div>
                  )
                })}
              </div>
            )}
          </div>
          <footer className="border-t border-[#dbe3e8] px-5 py-4 text-[10px] text-[#69717b]">
            <div>MedNeXt-S fold 0 · BTCV · server GPU</div>
            <div className="mt-4 border-t border-[#dbe3e8] pt-3 leading-relaxed">
              <div className="font-bold uppercase tracking-[0.14em] text-[#7d8996]">Made by:</div>
              <div className="mt-1 font-bold">Thai Ngoc Tam Nhu - 24110117</div>
              <div className="font-bold">Duong Duy Vinh - 24110148</div>
            </div>
          </footer>
        </aside>

        <section className="relative flex min-w-0 flex-1 flex-col">
          {!hasCases ? (
            <div className="flex flex-1 items-center justify-center p-8">
              <button
                type="button"
                onClick={() => setUploadOpen(true)}
                onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => { event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files) }}
                className={`group flex min-h-[348px] w-full max-w-[720px] flex-col items-center justify-center rounded-xl border border-dashed px-8 text-center transition-colors ${isDragging ? 'border-cyan-300 bg-cyan-300/[0.06]' : 'border-[#cbd8df] bg-[#f9fbfc] hover:border-[#647184] hover:bg-[#f1f6f8]'}`}
              >
                <span className="mb-6 flex size-16 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/[0.08] text-cyan-600 transition-transform group-hover:scale-105"><FileUp className="size-7" aria-hidden="true" /></span>
                <span className="text-[15px] font-medium text-[#3d4650]">Drag &amp; drop <span className="text-cyan-600">.nii / .nii.gz</span> files here</span>
                <span className="mt-2 text-[13px] text-[#697581]">or click to choose a CT volume</span>
                <span className="mt-8 flex items-center gap-2 text-[11px] text-[#4f5b67]"><CheckCircle2 className="size-3.5 text-emerald-500/80" /> Sent to the MedNeXt server for BTCV segmentation</span>
              </button>
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex min-h-14 flex-wrap items-center justify-between gap-3 border-b border-[#dbe3e8] px-6 py-2 text-xs text-[#7d8996]">
                <div className="min-w-0 truncate">
                  Viewer workspace <span className="mx-2 text-[#37414b]">/</span> {selectedCase?.name}
                </div>
                {selectedCase?.status === 'done' && selectedCase.jobId && (
                  <div className="flex shrink-0 items-center gap-2">
                    <button
                      type="button"
                      onClick={() => addGtRef.current?.click()}
                      disabled={Boolean(selectedCase.gtBusy)}
                      className="rounded-md border border-[#dbe3e8] bg-white px-2.5 py-1.5 text-[11px] font-semibold text-[#3d4650] hover:bg-[#f1f6f8] disabled:opacity-50"
                    >
                      {selectedCase.hasGt ? 'Replace Ground Truth' : '+ Add Ground Truth'}
                    </button>
                    {selectedCase.cached && (
                      <span className="rounded-full bg-cyan-50 px-2.5 py-1 text-[11px] font-semibold text-cyan-700">⚡ Loaded from Cache</span>
                    )}
                    {selectedCase.hasGt && typeof selectedCase.meanDice === 'number' && (
                      <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-semibold text-emerald-700">
                        Mean Dice {selectedCase.meanDice.toFixed(1)}%
                      </span>
                    )}
                    <span className="tabular-nums">Inference {selectedCase.processingTime?.toFixed(1)}s</span>
                    <a
                      href={selectedCase.downloadUrl}
                      download
                      className="rounded-md bg-[#647184] px-3 py-1.5 font-semibold text-white hover:bg-[#566276]"
                    >
                      Download .nii.gz
                    </a>
                  </div>
                )}
              </div>
              <input
                ref={addGtRef}
                type="file"
                accept=".nii,.nii.gz,.gz"
                className="sr-only"
                onChange={(event) => {
                  const file = event.target.files?.[0]
                  if (file) void addGroundTruth(file)
                  event.target.value = ''
                }}
              />

              {(showPhase || showRecover) && selectedCase && (
                <div className="flex flex-1 flex-col items-center justify-center gap-3 text-sm text-[#65717d]">
                  <LoaderCircle className="size-6 animate-spin text-cyan-600" />
                  <p>{showRecover ? phaseMessage('downloading') : phaseMessage(selectedCase.status)}</p>
                  {selectedCase.status === 'processing' && (
                    <p className="max-w-md text-center text-xs text-[#7d8996]">A cached CT returns immediately. A new volume still runs MedNeXt-S on the GPU.</p>
                  )}
                </div>
              )}

              {selectedCase?.status === 'error' && (
                <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
                  <p className="max-w-md text-sm text-red-500">{selectedCase.errorMessage || 'Segmentation failed.'}</p>
                  <Button onClick={() => uploadAndSegment(selectedCase.file, selectedCase.id, selectedCase.gtFile)} size="sm" className="h-8 rounded-md bg-[#647184] px-3 text-[12px] font-semibold text-white hover:bg-[#566276]">
                    Retry
                  </Button>
                </div>
              )}

              {selectedCase?.status === 'done' && selectedCase.jobId && bundle && (
                <CaseViewer
                  key={selectedCase.jobId}
                  volume={bundle}
                  organs={selectedCase.organs ?? []}
                  hasGt={Boolean(selectedCase.hasGt && bundle.gt)}
                  meanDice={selectedCase.meanDice}
                  initialSlice={sliceMemory.current[selectedCase.jobId] ?? Math.floor((bundle.shape[0] - 1) / 2)}
                  onSliceChange={(value) => {
                    if (selectedCase.jobId) sliceMemory.current[selectedCase.jobId] = value
                  }}
                  hotkeys={!uploadOpen}
                  gtBusy={selectedCase.gtBusy ?? null}
                />
              )}
            </div>
          )}
          {notice && <div role="status" className="absolute bottom-6 right-6 z-40 rounded-lg border border-red-400/20 bg-[#221416] px-4 py-3 text-xs text-red-200 shadow-xl">{notice}</div>}
        </section>
      </div>

      {uploadOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#1c2830]/40 p-6" onClick={closeUpload}>
          <div
            role="dialog"
            aria-labelledby="upload-title"
            className="w-full max-w-lg rounded-2xl border border-[#dbe3e8] bg-[#f9fbfc] p-6 shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="upload-title" className="text-[15px] font-semibold text-[#29313b]">New case</h2>
                <p className="mt-1 text-[12px] text-[#697581]">Choose a CT volume in .nii or .nii.gz.</p>
              </div>
              <button type="button" aria-label="Close" onClick={closeUpload} className="text-[#66717d] hover:text-[#29313b]">
                <X className="size-4" />
              </button>
            </div>

            <div className="mt-5 space-y-3">
              <label className="block rounded-xl border border-dashed border-[#cbd8df] bg-white px-4 py-4 text-left">
                <span className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#7d8996]">CT Volume</span>
                <span className="mt-1 block truncate text-[13px] text-[#29313b]">{pendingCt ? pendingCt.name : 'Choose .nii or .nii.gz'}</span>
                <input
                  ref={ctPickerRef}
                  type="file"
                  accept=".nii,.nii.gz,.gz"
                  className="sr-only"
                  onChange={(event) => {
                    const file = event.target.files?.[0] ?? null
                    if (file && !isNiftiFile(file.name)) {
                      showNotice('Only .nii and .nii.gz files are supported.')
                      event.target.value = ''
                      return
                    }
                    setPendingCt(file)
                  }}
                />
              </label>
            </div>

            <div className="mt-5 flex justify-end gap-2">
              <button type="button" onClick={closeUpload} className="rounded-md border border-[#dbe3e8] bg-white px-3 py-2 text-[12px] font-semibold text-[#3d4650] hover:bg-[#f1f6f8]">
                Cancel
              </button>
              <button type="button" onClick={submitUpload} disabled={!pendingCt} className="rounded-md bg-[#647184] px-3 py-2 text-[12px] font-semibold text-white hover:bg-[#566276] disabled:opacity-40">
                Start segmentation
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
