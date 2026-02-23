from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response  # <-- ДОБАВЬ Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import shutil
import zipfile
import uuid
from typing import List, Optional
import logging

# Импортируем модули
import video_downloader
import photo_downloader
from batch_processor import get_batch_processor
from watermark_remover import get_remover

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="TikTok Downloader")
templates = Jinja2Templates(directory="templates")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создаем папки
os.makedirs("downloads/videos", exist_ok=True)
os.makedirs("downloads/photos", exist_ok=True)
os.makedirs("downloads/batch", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("masks", exist_ok=True)

# Инициализация обработчиков
batch_processor = get_batch_processor()
watermark_remover = get_remover(mask_dir="masks/")

# ==================== ГЛАВНЫЕ СТРАНИЦЫ ====================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/photos", response_class=HTMLResponse)
async def photos_page(request: Request):
    """Страница выбора фото"""
    return templates.TemplateResponse("photos.html", {"request": request})

@app.get("/remove-watermark", response_class=HTMLResponse)
async def remove_watermark_page(request: Request):
    """Страница удаления водяного знака (1 фото)"""
    return templates.TemplateResponse("watermark.html", {"request": request})

@app.get("/batch-watermark", response_class=HTMLResponse)
async def batch_watermark_page(request: Request):
    """Страница пакетной обработки ZIP архива"""
    return templates.TemplateResponse("batch_watermark.html", {"request": request})

# ==================== API ДЛЯ TIKTOK ====================

@app.post("/analyze")
async def analyze_url(url: str = Form(...)):
    """Анализирует ссылку и определяет тип контента"""
    try:
        # Разрешаем короткую ссылку если нужно
        if 'vt.tiktok.com' in url:
            url = await video_downloader.resolve_short_url(url)
        
        # Проверяем, фото или видео
        if photo_downloader.is_photo_url(url):
            return {'type': 'photos', 'redirect': '/photos'}
        
        # Проверяем видео
        video_info = video_downloader.get_video_info(url)
        if video_info.get('is_video'):
            return video_info
        
        return {'type': 'unknown', 'error': 'Не удалось определить тип контента'}
        
    except Exception as e:
        logger.error(f"Ошибка analyze_url: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/analyze/photos")
async def analyze_photos(url: str = Form(...)):
    """Анализирует ссылку и возвращает список фото для выбора"""
    try:
        photo_urls = await photo_downloader.get_photo_urls(url)
        
        if not photo_urls:
            return JSONResponse(
                status_code=404,
                content={"error": "Не найдено фото по этой ссылке"}
            )
        
        return {
            'type': 'photos',
            'images': photo_urls,
            'count': len(photo_urls),
            'title': 'TikTok Photos',
            'author': 'Unknown'
        }
        
    except Exception as e:
        logger.error(f"Ошибка analyze_photos: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/download/video")
async def download_video(url: str = Form(...)):
    """Скачивает видео"""
    try:
        # Разрешаем короткую ссылку если нужно
        if 'vt.tiktok.com' in url:
            url = await video_downloader.resolve_short_url(url)
        
        file_path = await asyncio.to_thread(video_downloader.download_video, url)
        
        if not os.path.exists(file_path):
            return JSONResponse(
                status_code=404,
                content={"error": "Видео не найдено"}
            )
            
        return FileResponse(
            path=file_path,
            media_type="video/mp4",
            filename="tiktok_video.mp4"
        )
        
    except Exception as e:
        logger.error(f"Ошибка download_video: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/download/photos")
async def download_photos(image_urls: List[str] = Form(...), download_type: str = Form(...)):
    """Скачивает выбранные фото"""
    try:
        # Скачиваем выбранные фото
        photo_paths = await photo_downloader.download_selected_photos(image_urls)
        
        if not photo_paths:
            return JSONResponse(
                status_code=404,
                content={"error": "Не удалось скачать фото"}
            )
        
        # Если выбран режим ZIP
        if download_type == 'zip':
            zip_filename = f"tiktok_photos_{uuid.uuid4()}.zip"
            zip_path = os.path.join("downloads/photos", zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for file_path in photo_paths:
                    zipf.write(file_path, os.path.basename(file_path))
            
            return FileResponse(
                path=zip_path,
                media_type="application/zip",
                filename="tiktok_photos.zip"
            )
        
        # Если выбран режим "по одному"
        else:
            if len(photo_paths) == 1:
                filename = os.path.basename(photo_paths[0])
                return FileResponse(
                    path=photo_paths[0],
                    media_type="image/jpeg",
                    filename=filename
                )
            
            # Если несколько - возвращаем список файлов
            file_list = []
            for i, path in enumerate(photo_paths):
                filename = os.path.basename(path)
                new_filename = f"tiktok_photo_{i+1}.jpg"
                new_path = os.path.join("downloads/photos", new_filename)
                shutil.copy2(path, new_path)
                file_list.append({
                    'filename': new_filename,
                    'url': f'/files/{new_filename}',
                    'original': filename
                })
            
            return {
                'type': 'multiple',
                'files': file_list,
                'count': len(file_list)
            }
        
    except Exception as e:
        logger.error(f"Ошибка download_photos: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

# ==================== API ДЛЯ ВОДЯНЫХ ЗНАКОВ ====================

@app.post("/api/remove-watermark")
async def api_remove_watermark(
    file: UploadFile = File(...),
    size: str = Form("auto")
):
    """Удаляет водяной знак Gemini из загруженного изображения"""
    try:
        # Проверяем тип файла
        if not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Файл должен быть изображением"}
            )
        
        # Читаем файл
        contents = await file.read()
        
        # Определяем размер
        force_size = None if size == "auto" else size
        
        # Удаляем водяной знак
        clean_bytes = watermark_remover.remove_watermark(contents, force_size=force_size)
        
        # Возвращаем очищенное изображение - используем Response, а не FileResponse
        filename = f"clean_{file.filename}"
        if not filename.lower().endswith('.png'):
            filename = filename.rsplit('.', 1)[0] + '.png'
        
        # ВАЖНО: используем Response с content_type image/png
        return Response(
            content=clean_bytes,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Ошибка remove-watermark: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
@app.post("/api/batch-process")
async def batch_process(
    file: UploadFile = File(...),
    size: str = Form("auto"),
    workers: int = Form(4),
    keep_structure: bool = Form(True)
):
    """Загружает ZIP архив и обрабатывает все изображения"""
    temp_path = None
    try:
        # Проверяем тип файла
        if not file.filename.endswith('.zip'):
            return JSONResponse(
                status_code=400,
                content={"error": "Файл должен быть ZIP архивом"}
            )
        
        # Сохраняем загруженный файл
        file_id = str(uuid.uuid4())[:8]
        temp_path = os.path.join("uploads", f"upload_{file_id}.zip")
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Файл сохранен: {temp_path}")
        
        # Обрабатываем
        output_zip, stats = batch_processor.process_batch(
            temp_path,
            size=size,
            max_workers=workers,
            keep_structure=keep_structure
        )
        
        logger.info(f"Обработка завершена: {output_zip}")
        
        # Возвращаем результат
        return FileResponse(
            path=output_zip,
            media_type="application/zip",
            filename=f"processed_{file_id}.zip"
        )
        
    except Exception as e:
        logger.error(f"Ошибка batch-process: {e}")
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )
    finally:
        # Очищаем загруженный файл
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/api/batch-status/{batch_id}")
async def batch_status(batch_id: str):
    """Получает статус обработки по ID пакета"""
    status = batch_processor.get_batch_status(batch_id)
    if status:
        return JSONResponse(content=status)
    return JSONResponse(
        status_code=404,
        content={"error": "Пакет не найден"}
    )

@app.get("/files/{filename}")
async def get_file(filename: str):
    """Отдаёт файл по имени"""
    # Ищем в разных папках
    possible_paths = [
        os.path.join("downloads/photos", filename),
        os.path.join("downloads/batch", filename),
        os.path.join("uploads", filename)
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return FileResponse(
                path=path,
                media_type="image/jpeg" if filename.endswith(('.jpg', '.jpeg')) else "application/octet-stream",
                filename=filename
            )
    
    return JSONResponse(
        status_code=404,
        content={"error": "Файл не найден"}
    )

# ==================== СТАТИКА И ОЧИСТКА ====================

@app.on_event("startup")
async def startup():
    """При запуске создаем папки и чистим старые файлы"""
    folders = [
        "downloads/videos", 
        "downloads/photos", 
        "downloads/batch",
        "uploads",
        "masks"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        
        # Очищаем файлы старше 1 часа
        if os.path.exists(folder):
            import time
            current_time = time.time()
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path):
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age > 3600:  # 1 час
                            os.unlink(file_path)
                            logger.info(f"Удален старый файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка удаления {file_path}: {e}")
    
    logger.info("Сервер запущен и готов к работе")

@app.on_event("shutdown")
async def shutdown():
    """При остановке очищаем все временные файлы"""
    try:
        shutil.rmtree("uploads", ignore_errors=True)
        shutil.rmtree("downloads/batch", ignore_errors=True)
        logger.info("Временные файлы очищены")
    except:
        pass

# Для локального запуска
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
