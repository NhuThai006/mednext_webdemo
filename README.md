# MedNeXt Web Demo

Hệ thống Web Demo phân đoạn tạng ổ bụng (Abdominal Organ Segmentation) từ ảnh CT dạng NIfTI (`.nii` / `.nii.gz`) sử dụng mô hình MedNeXt (nnU-Net backbone) huấn luyện trên bộ dữ liệu BTCV (14 classes: Background + 13 organs).

## 🚀 Cấu trúc dự án

- **`backend/`**: FastAPI backend xử lý API upload, inference model MedNeXt, sinh kết quả segmentation và render trực quan lát cắt CT.
- **`frontend/`**: Next.js 16 (App Router) + Tailwind CSS + Lucide Icons cung cấp giao diện người dùng tương tác trực tiếp.
- **`model/`**: Kiến trúc mô hình MedNeXt và nnUNet-MedNeXt wrapper.

---

## 🛠️ Hướng dẫn khởi chạy

### 1. Khởi động Backend (FastAPI)
```bash
conda activate dip_web
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```
Backend API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Khởi động Frontend (Next.js)
Mở một terminal mới:
```bash
cd frontend
npm install  # hoặc pnpm install
npm run dev
```
Mở trình duyệt tại [http://localhost:3000](http://localhost:3000)

---

## 📋 Yêu cầu dữ liệu & Checkpoints
Mô hình yêu cầu checkpoint đã huấn luyện đặt tại:
`model/data/nnUNet_trained_models/nnUNet/3d_fullres/Task017_AbdominalOrganSegmentation/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/`
