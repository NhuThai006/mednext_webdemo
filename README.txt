Khi muốn dùng lại, mở hai terminal:


conda activate dip_web
cd D:\university\nckh2627\gd2_finetune\code\MedNeXt-webdemo\backend
uvicorn main:app --host 0.0.0.0 --port 8000

cd D:\university\nckh2627\gd2_finetune\code\MedNeXt-webdemo\frontend
npm run dev
Sau đó mở http://localhost:3000.