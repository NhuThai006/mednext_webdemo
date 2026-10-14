'use client'

import { useRef, useState } from 'react'
import { CheckCircle2, FileUp, FolderOpen, LoaderCircle, Upload, X } from 'lucide-react'
import { Button } from '@/components/ui/button'

const acceptedExtensions = ['.nii', '.nii.gz']

// Đổi thành http://<IP_SERVER>:8000 nếu không dùng SSH tunnel.
const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL ?? 'http://localhost:8000'

type OrganInfo = {
  id: number
  name: string
  color: string
  present: boolean
}

type CaseStatus = 'uploading' | 'processing' | 'done' | 'error'

type CaseFile = {
  id: string
  file: File
  name: string
  size: string
  status: CaseStatus
  jobId?: string
  downloadUrl?: string
  totalSlices?: number
  currentSlice?: number
  organs?: OrganInfo[]
  processingTime?: number
  errorMessage?: string
}

type SegmentResponse = {
  detail?: string
  job_id?: string
  total_slices?: number
  organs_detected?: OrganInfo[]
  processing_time_seconds?: number
}

function isNiftiFile(name: string) {
  return acceptedExtensions.some((extension) => name.toLowerCase().endsWith(extension))
}

function formatFileSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function statusLabel(status: CaseStatus) {
  if (status === 'done') return 'Done'
  if (status === 'error') return 'Error'
  if (status === 'uploading') return 'Uploading'
  return 'Processing'
}

function statusClass(status: CaseStatus) {
  if (status === 'done') return 'text-emerald-600'
  if (status === 'error') return 'text-red-500'
  return 'text-cyan-600'
}

export default function Page() {
  const inputRef = useRef<HTMLInputElement>(null)
  const [cases, setCases] = useState<CaseFile[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [notice, setNotice] = useState('')

  const selectedCase = cases.find((item) => item.id === selectedId) ?? cases[0] ?? null
  const hasCases = cases.length > 0

  async function uploadAndSegment(file: File, caseId: string) {
    const formData = new FormData()
    formData.append('file', file)

    setCases((prev) => prev.map((item) => (
      item.id === caseId ? { ...item, status: 'processing', errorMessage: undefined } : item
    )))

    try {
      const res = await fetch(`${BACKEND}/api/segment`, { method: 'POST', body: formData })
      const data = await res.json().catch(() => ({})) as SegmentResponse
      const jobId = data.job_id
      const totalSlices = data.total_slices
      if (!res.ok || !jobId || totalSlices == null) {
        throw new Error(data.detail || 'Segmentation failed')
      }

      setCases((prev) => prev.map((item) => (
        item.id === caseId ? {
          ...item,
          status: 'done',
          jobId,
          downloadUrl: `${BACKEND}/api/result/${jobId}`,
          totalSlices,
          currentSlice: Math.floor(totalSlices / 2),
          organs: data.organs_detected,
          processingTime: data.processing_time_seconds,
        } : item
      )))
    } catch (err: unknown) {
      const raw = err instanceof Error ? err.message : 'Unknown error'
      const msg = raw === 'Failed to fetch'
        ? `Cannot reach the segmentation server at ${BACKEND}.`
        : raw
      setCases((prev) => prev.map((item) => (
        item.id === caseId ? { ...item, status: 'error', errorMessage: msg } : item
      )))
    }
  }

  function addFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList)
    const invalid = files.find((file) => !isNiftiFile(file.name))
    if (invalid) {
      setNotice('Only .nii and .nii.gz files are supported.')
      window.setTimeout(() => setNotice(''), 3600)
    }
    const validFiles = files.filter((file) => isNiftiFile(file.name))
    if (!validFiles.length) return

    const nextCases: CaseFile[] = validFiles.map((file) => ({
      id: `${file.name}-${file.lastModified}-${Math.random()}`,
      file,
      name: file.name,
      size: formatFileSize(file.size),
      status: 'uploading',
    }))

    setCases((current) => [...current, ...nextCases])
    setSelectedId((current) => current ?? nextCases[0].id)
    nextCases.forEach((item) => {
      void uploadAndSegment(item.file, item.id)
    })
    if (inputRef.current) inputRef.current.value = ''
  }

  function removeCase(id: string) {
    setCases((current) => current.filter((entry) => entry.id !== id))
    setSelectedId((current) => (current === id ? null : current))
  }

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
            <Button onClick={() => inputRef.current?.click()} size="sm" className="h-8 gap-2 rounded-md bg-[#647184] px-3 text-[12px] font-semibold text-white hover:bg-[#566276]">
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
                          {item.status === 'processing' || item.status === 'uploading'
                            ? <LoaderCircle className="mt-0.5 size-4 shrink-0 animate-spin text-cyan-500" />
                            : <FileUp className="mt-0.5 size-4 shrink-0 text-cyan-600" />}
                          <span className="min-w-0 flex-1 truncate text-[12px]">{item.name}</span>
                        </button>
                        <button onClick={() => removeCase(item.id)} aria-label={`Remove ${item.name}`} className="text-[#66717d] hover:text-[#29313b]">
                          <X className="size-3.5" />
                        </button>
                      </div>
                      <div className="mt-2 flex items-center gap-2 pl-6 text-[10px] text-[#6f7b88]">
                        <span>{item.size}</span>
                        <span className="text-[#39424c]">·</span>
                        <span className={statusClass(item.status)}>{statusLabel(item.status)}</span>
                      </div>
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
          <input ref={inputRef} type="file" multiple accept=".nii,.nii.gz,.gz" className="sr-only" onChange={(event) => event.target.files && addFiles(event.target.files)} />
          {!hasCases ? (
            <div className="flex flex-1 items-center justify-center p-8">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(event) => { event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files) }}
                className={`group flex min-h-[348px] w-full max-w-[720px] flex-col items-center justify-center rounded-xl border border-dashed px-8 text-center transition-colors ${isDragging ? 'border-cyan-300 bg-cyan-300/[0.06]' : 'border-[#cbd8df] bg-[#f9fbfc] hover:border-[#647184] hover:bg-[#f1f6f8]'}`}
              >
                <span className="mb-6 flex size-16 items-center justify-center rounded-2xl border border-cyan-300/20 bg-cyan-300/[0.08] text-cyan-600 transition-transform group-hover:scale-105"><FileUp className="size-7" aria-hidden="true" /></span>
                <span className="text-[15px] font-medium text-[#3d4650]">Drag &amp; drop <span className="text-cyan-600">.nii / .nii.gz</span> files here</span>
                <span className="mt-2 text-[13px] text-[#697581]">or click to browse</span>
                <span className="mt-8 flex items-center gap-2 text-[11px] text-[#4f5b67]"><CheckCircle2 className="size-3.5 text-emerald-500/80" /> Sent to the MedNeXt server for BTCV segmentation</span>
              </button>
            </div>
          ) : (
            <div className="flex min-h-0 flex-1 flex-col">
              <div className="flex h-14 items-center border-b border-[#dbe3e8] px-6 text-xs text-[#7d8996]">
                Viewer workspace <span className="mx-2 text-[#37414b]">/</span> {selectedCase?.name}
              </div>

              {selectedCase && (selectedCase.status === 'processing' || selectedCase.status === 'uploading') && (
                <div className="flex flex-1 flex-col items-center justify-center gap-3 text-sm text-[#65717d]">
                  <LoaderCircle className="size-6 animate-spin text-cyan-600" />
                  <p>Uploading and segmenting on the server.</p>
                  <p className="text-xs text-[#7d8996]">MedNeXt-S fold 0 usually takes a few minutes.</p>
                </div>
              )}

              {selectedCase?.status === 'error' && (
                <div className="flex flex-1 flex-col items-center justify-center gap-3 px-6 text-center">
                  <p className="max-w-md text-sm text-red-500">{selectedCase.errorMessage || 'Segmentation failed.'}</p>
                  <Button onClick={() => uploadAndSegment(selectedCase.file, selectedCase.id)} size="sm" className="h-8 rounded-md bg-[#647184] px-3 text-[12px] font-semibold text-white hover:bg-[#566276]">
                    Retry
                  </Button>
                </div>
              )}

              {selectedCase?.status === 'done' && selectedCase.jobId && (
                <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-auto p-6">
                  <div className="overflow-hidden rounded-xl border border-[#dbe3e8] bg-black">
                    <img
                      key={`${selectedCase.jobId}-${selectedCase.currentSlice}`}
                      src={`${BACKEND}/api/preview/${selectedCase.jobId}?slice=${selectedCase.currentSlice ?? 0}&alpha=0.5`}
                      alt="Segmentation preview"
                      className="max-h-[68vh] w-full object-contain"
                    />
                  </div>

                  <div className="flex items-center gap-3">
                    <span className="w-16 shrink-0 text-xs text-[#7d8996]">Axial</span>
                    <input
                      type="range"
                      min={0}
                      max={(selectedCase.totalSlices ?? 1) - 1}
                      value={selectedCase.currentSlice ?? 0}
                      onChange={(event) => {
                        const val = Number(event.target.value)
                        setCases((prev) => prev.map((item) => (
                          item.id === selectedCase.id ? { ...item, currentSlice: val } : item
                        )))
                      }}
                      className="flex-1 accent-cyan-500"
                    />
                    <span className="w-20 shrink-0 text-right text-xs text-[#7d8996]">
                      {(selectedCase.currentSlice ?? 0) + 1} / {selectedCase.totalSlices}
                    </span>
                  </div>

                  <div className="rounded-lg border border-[#dbe3e8] bg-[#f9fbfc] p-3">
                    <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#7d8996]">
                      Detected Organs (BTCV)
                    </div>
                    <div className="flex flex-wrap gap-x-4 gap-y-2">
                      {selectedCase.organs?.map((organ) => (
                        <div key={organ.id} className="flex items-center gap-1.5">
                          <span
                            className="inline-block size-3 shrink-0 rounded-sm border border-black/10"
                            style={{ backgroundColor: organ.color, opacity: organ.present ? 1 : 0.25 }}
                          />
                          <span className={`text-[11px] ${organ.present ? 'text-[#3d4650]' : 'text-[#9aa6b2] line-through'}`}>
                            {organ.name}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex items-center justify-between text-xs text-[#7d8996]">
                    <span>{selectedCase.processingTime?.toFixed(1)}s · MedNeXt-S fold_0 · BTCV</span>
                    <a
                      href={selectedCase.downloadUrl}
                      download
                      className="flex items-center gap-1.5 rounded-md bg-[#647184] px-3 py-1.5 text-white hover:bg-[#566276]"
                    >
                      Download .nii.gz
                    </a>
                  </div>
                </div>
              )}
            </div>
          )}
          {notice && <div role="status" className="absolute bottom-6 right-6 rounded-lg border border-red-400/20 bg-[#221416] px-4 py-3 text-xs text-red-200 shadow-xl">{notice}</div>}
        </section>
      </div>
    </main>
  )
}
