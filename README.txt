Khi muốn dùng lại, mở hai terminal tại thư mục MedNeXt-webdemo.

File môi trường frontend (chỉ cần làm một lần):

copy frontend\.env.example frontend\.env.local

Nội dung frontend\.env.local:

NEXT_PUBLIC_BACKEND_URL=http://localhost:8000

Sửa URL xong thì tắt và chạy lại frontend.


Terminal 1:

conda activate dip_web
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000

Terminal 2:

cd frontend
pnpm dev

Sau đó mở http://localhost:3000.
API: http://127.0.0.1:8000/api/health
