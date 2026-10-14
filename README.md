# MedNeXt Web Demo

Web demo phân đoạn tạng ổ bụng từ CT NIfTI (`.nii` / `.nii.gz`) bằng MedNeXt (nnU-Net), huấn luyện trên BTCV (nền + 13 tạng).

- `backend/`: FastAPI, inference, cache kết quả, volume 3D cho trình duyệt.
- `frontend/`: Next.js 16, Tailwind CSS. Vẽ lát cắt bằng canvas.
- `model/`: kiến trúc MedNeXt và checkpoint. Không sửa khi chỉ chạy demo.

Backend tự gán `nnUNet_raw_data_base`, `nnUNet_preprocessed` và `RESULTS_FOLDER` trong `backend/config.py`. Không cần file env cho Python.

## Yêu cầu

- Conda env `dip_web` (Python, PyTorch CUDA, các gói trong `backend/requirements.txt`).
- Node.js và pnpm hoặc npm.
- GPU NVIDIA. Máy demo dùng RTX 3050 6GB.
- Checkpoint tại:

`model/data/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/`

File trọng số trong fold là `model_best`.

## File môi trường

Frontend đọc URL backend từ `frontend/.env.local`.

```powershell
copy frontend\.env.example frontend\.env.local
```

Nội dung mặc định, khi API chạy trên cùng máy:

```env
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

Nếu backend ở máy khác, đổi thành `http://<IP>:8000`. Sau khi sửa file này, tắt `next dev` rồi chạy lại. Trình duyệt không nhận giá trị mới nếu chỉ lưu file khi server frontend vẫn đang chạy.

`.env.local` không được commit. Mẫu nằm ở `frontend/.env.example`.

## Chạy trên máy

Mở hai terminal tại thư mục gốc repo `MedNeXt-webdemo`.

**Terminal 1 — API**

```powershell
conda activate dip_web
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

Đợi dòng `Application startup complete`. Kiểm tra [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health). Kết quả đúng có `"status": "ok"` và `"device": "cuda"`. Tài liệu API: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

Lệnh phải chạy trong `backend/`. Uvicorn import `main:app` từ thư mục đó.

**Terminal 2 — giao diện**

```powershell
cd frontend
pnpm install
pnpm dev
```

Có thể dùng `npm install` rồi `npm run dev`. Mở [http://localhost:3000](http://localhost:3000).

Upload một file CT `.nii` hoặc `.nii.gz`. Lần đầu chạy model có thể mất vài phút. Upload lại đúng file đó sẽ lấy kết quả từ `backend/temp/cache/`.

Nếu trang báo không kết nối được `http://localhost:8000`, API ở terminal 1 chưa chạy hoặc đang tắt. Bật lại API rồi bấm Retry.

Muốn máy khác trong mạng gọi API, đổi host thành `0.0.0.0` và trỏ `NEXT_PUBLIC_BACKEND_URL` tới IP của máy chạy backend.
