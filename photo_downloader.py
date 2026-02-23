import os
import uuid
import asyncio
import httpx
import aiofiles
import subprocess
import re
from typing import List

PHOTOS_FOLDER = "downloads/photos"
os.makedirs(PHOTOS_FOLDER, exist_ok=True)

async def resolve_short_url(url: str) -> str:
    """Разрешает короткую ссылку vt.tiktok.com до полной"""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.head(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            return str(response.url)
    except Exception as e:
        print(f"Ошибка разрешения короткой ссылки: {e}")
        return url

def is_photo_url(url: str) -> bool:
    """Проверяет, что ссылка ведёт на фото/слайд-шоу"""
    url_lower = url.lower()
    return '/photo/' in url_lower or '/photos/' in url_lower

async def get_photo_urls(url: str) -> List[str]:
    """
    Получает список URL фото без скачивания
    """
    # Сначала разрешаем короткую ссылку если нужно
    if 'vt.tiktok.com' in url:
        url = await resolve_short_url(url)
    
    photo_urls = []
    
    # Способ 1: через gallery-dl
    try:
        cmd = [
            'gallery-dl',
            '--get-urls',
            '--no-download',
            url
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            urls = stdout.decode().strip().split('\n')
            photo_urls = [u for u in urls if u.startswith('http') and any(ext in u.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp'])]
            if photo_urls:
                return photo_urls
    except:
        pass
    
    # Способ 2: парсинг HTML
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            html = response.text
            
            patterns = [
                r'https?://[^\s"\']+\.(?:jpg|jpeg|png|webp)[^\s"\']*',
                r'content="(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r'src="(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"',
                r'data-src="(https?://[^"]+\.(?:jpg|jpeg|png|webp)[^"]*)"'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html, re.IGNORECASE)
                photo_urls.extend(matches)
            
            photo_urls = list(set(photo_urls))
            photo_urls = [u for u in photo_urls if 'avatar' not in u and 'logo' not in u]
            
            if photo_urls:
                return photo_urls[:20]
    except Exception as e:
        print(f"Ошибка парсинга HTML: {e}")
    
    return []

async def download_selected_photos(image_urls: List[str]) -> List[str]:
    """
    Скачивает выбранные фото и возвращает пути к файлам
    """
    downloaded_files = []
    
    for i, img_url in enumerate(image_urls):
        try:
            # Определяем расширение
            ext_match = re.search(r'\.(jpg|jpeg|png|webp)', img_url.lower())
            ext = ext_match.group(1) if ext_match else 'jpg'
            if ext == 'jpeg':
                ext = 'jpg'
            
            unique_name = f"{uuid.uuid4()}.{ext}"
            file_path = os.path.join(PHOTOS_FOLDER, unique_name)
            
            # Скачиваем с повторными попытками
            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        }
                        response = await client.get(img_url, headers=headers)
                        
                        if response.status_code == 200:
                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(response.content)
                            downloaded_files.append(file_path)
                            break
                except Exception as e:
                    if attempt == 2:
                        print(f"Ошибка скачивания {img_url}: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            print(f"Ошибка обработки фото {i}: {e}")
    
    return downloaded_files
