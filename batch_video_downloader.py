"""
Batch Video Downloader - Пакетное скачивание видео из TikTok
Автор: Tokiyashe
Дата: 2026
"""

import os
import zipfile
import shutil
import json
import asyncio
import uuid
import logging
import re
import time
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import video_downloader

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BatchVideoDownloader:
    """
    Пакетное скачивание видео из TikTok по списку ссылок
    """
    
    def __init__(self, download_dir: str = "downloads/batch_videos"):
        """
        Инициализация обработчика
        
        Args:
            download_dir: директория для сохранения видео и архивов
        """
        self.download_dir = download_dir
        self.temp_dir = os.path.join(download_dir, "temp")
        
        # Создаем необходимые папки
        os.makedirs(self.download_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"BatchVideoDownloader инициализирован")
        logger.info(f"  Download dir: {download_dir}")
        logger.info(f"  Temp dir: {self.temp_dir}")
    
    def validate_url(self, url: str) -> bool:
        """
        Проверяет, является ли строка валидной ссылкой TikTok
        """
        url = url.strip()
        if not url:
            return False
        
        # Проверяем наличие tiktok.com в ссылке
        if 'tiktok.com' not in url:
            return False
        
        # Проверяем формат URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False
        
        return True
    
    def extract_urls_from_text(self, text: str) -> List[str]:
        """
        Извлекает все ссылки из текста
        """
        # Регулярное выражение для поиска URL
        url_pattern = r'https?://[^\s<>"\'(){}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        
        # Фильтруем только ссылки TikTok
        tiktok_urls = []
        for url in urls:
            if 'tiktok.com' in url:
                # Очищаем URL от лишних параметров
                clean_url = url.split('?')[0]
                tiktok_urls.append(clean_url)
        
        return list(set(tiktok_urls))  # Убираем дубликаты
    
    async def download_single_video(self, 
                                   url: str, 
                                   output_dir: str,
                                   index: int,
                                   quality: str = "high") -> Dict:
        """
        Скачивает одно видео
        
        Args:
            url: ссылка на видео
            output_dir: папка для сохранения
            index: порядковый номер
            quality: качество видео ('high' или 'low')
            
        Returns:
            словарь с информацией о результате
        """
        try:
            logger.info(f"[{index}] Скачивание: {url}")
            start_time = time.time()
            
            # Разрешаем короткую ссылку если нужно
            if 'vt.tiktok.com' in url:
                resolved_url = await video_downloader.resolve_short_url(url)
                logger.info(f"[{index}] Разрешенная ссылка: {resolved_url}")
                url = resolved_url
            
            # Получаем информацию о видео
            video_info = video_downloader.get_video_info(url)
            
            if not video_info.get('is_video'):
                return {
                    'success': False,
                    'url': url,
                    'error': 'Это не видео или видео недоступно',
                    'index': index
                }
            
            # Скачиваем видео
            file_path = video_downloader.download_video(url)
            
            if not os.path.exists(file_path):
                return {
                    'success': False,
                    'url': url,
                    'error': 'Файл не был создан',
                    'index': index
                }
            
            # Генерируем имя для сохранения
            title = video_info.get('title', f'video_{index}')
            # Очищаем название от недопустимых символов
            title = re.sub(r'[\\/*?:"<>|]', "_", title)
            # Ограничиваем длину
            if len(title) > 50:
                title = title[:50]
            
            author = video_info.get('author', 'unknown')
            author = re.sub(r'[\\/*?:"<>|]', "_", author)
            
            # Формируем имя файла
            ext = os.path.splitext(file_path)[1]
            filename = f"{index:03d}_{author}_{title}{ext}"
            new_path = os.path.join(output_dir, filename)
            
            # Перемещаем файл
            shutil.move(file_path, new_path)
            
            process_time = time.time() - start_time
            
            return {
                'success': True,
                'url': url,
                'title': video_info.get('title', 'Unknown'),
                'author': video_info.get('author', 'Unknown'),
                'filename': filename,
                'file_path': new_path,
                'file_size': os.path.getsize(new_path),
                'duration': video_info.get('duration', 0),
                'time': round(process_time, 2),
                'index': index
            }
            
        except Exception as e:
            logger.error(f"[{index}] Ошибка скачивания {url}: {e}")
            return {
                'success': False,
                'url': url,
                'error': str(e),
                'index': index
            }
    
    async def download_batch(self, 
                           urls: List[str],
                           quality: str = "high",
                           max_workers: int = 3,
                           create_zip: bool = True,
                           zip_name: Optional[str] = None) -> Tuple[Optional[str], Dict]:
        """
        Скачивает несколько видео
        
        Args:
            urls: список ссылок на видео
            quality: качество видео
            max_workers: количество параллельных загрузок
            create_zip: создавать ли ZIP архив
            zip_name: имя ZIP архива (если None, генерируется автоматически)
            
        Returns:
            (путь к ZIP архиву или None, статистика)
        """
        # Создаем временную папку для этого пакета
        batch_id = str(uuid.uuid4())[:8]
        temp_output_dir = os.path.join(self.temp_dir, f"batch_{batch_id}")
        os.makedirs(temp_output_dir, exist_ok=True)
        
        stats = {
            'batch_id': batch_id,
            'total': len(urls),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'results': [],
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': None,
            'total_time': None,
            'quality': quality
        }
        
        try:
            start_total = time.time()
            
            # Валидация URL
            valid_urls = []
            invalid_urls = []
            
            for url in urls:
                if self.validate_url(url):
                    valid_urls.append(url)
                else:
                    invalid_urls.append(url)
            
            stats['skipped'] = len(invalid_urls)
            stats['total'] = len(valid_urls)
            
            if invalid_urls:
                logger.warning(f"Пропущено {len(invalid_urls)} невалидных ссылок")
            
            if not valid_urls:
                raise ValueError("Нет валидных ссылок для скачивания")
            
            logger.info(f"Начинаем скачивание {len(valid_urls)} видео")
            
            # Создаем задачи для параллельного скачивания
            tasks = []
            for i, url in enumerate(valid_urls, 1):
                task = self.download_single_video(
                    url, 
                    temp_output_dir, 
                    i,
                    quality
                )
                tasks.append(task)
            
            # Запускаем с ограничением параллельности
            semaphore = asyncio.Semaphore(max_workers)
            
            async def bounded_download(task):
                async with semaphore:
                    return await task
            
            # Выполняем все задачи
            results = await asyncio.gather(*[bounded_download(task) for task in tasks])
            
            # Собираем статистику
            for result in results:
                stats['results'].append(result)
                if result.get('success'):
                    stats['success'] += 1
                else:
                    stats['failed'] += 1
            
            stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            stats['total_time'] = round(time.time() - start_total, 2)
            
            # Создаем отчет
            report_path = os.path.join(temp_output_dir, "report.txt")
            self._generate_report(stats, report_path)
            
            # Создаем JSON отчет
            json_report_path = os.path.join(temp_output_dir, "report.json")
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
            
            # Создаем ZIP архив если нужно
            if create_zip:
                if zip_name is None:
                    timestamp = time.strftime('%Y%m%d_%H%M%S')
                    zip_name = f"tiktok_videos_{timestamp}.zip"
                elif not zip_name.endswith('.zip'):
                    zip_name += '.zip'
                
                zip_path = os.path.join(self.download_dir, zip_name)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Добавляем все видео
                    for result in stats['results']:
                        if result.get('success') and os.path.exists(result['file_path']):
                            zipf.write(result['file_path'], result['filename'])
                    
                    # Добавляем отчеты
                    if os.path.exists(report_path):
                        zipf.write(report_path, "report.txt")
                    if os.path.exists(json_report_path):
                        zipf.write(json_report_path, "report.json")
                
                logger.info(f"ZIP архив создан: {zip_path}")
                return zip_path, stats
            
            return None, stats
            
        except Exception as e:
            logger.error(f"Ошибка пакетного скачивания: {e}")
            raise
        finally:
            # Очищаем временную папку
            try:
                shutil.rmtree(temp_output_dir, ignore_errors=True)
            except:
                pass
    
    def _generate_report(self, stats: Dict, output_path: str):
        """Генерирует текстовый отчет"""
        lines = []
        lines.append("="*80)
        lines.append("ОТЧЕТ О ПАКЕТНОМ СКАЧИВАНИИ ВИДЕО")
        lines.append("="*80)
        lines.append(f"ID пакета: {stats['batch_id']}")
        lines.append(f"Начало: {stats['start_time']}")
        lines.append(f"Конец: {stats['end_time']}")
        lines.append(f"Общее время: {stats['total_time']} сек")
        lines.append(f"Качество: {stats['quality']}")
        lines.append("-"*80)
        lines.append(f"Всего ссылок: {stats['total']}")
        lines.append(f"Успешно скачано: {stats['success']}")
        lines.append(f"Ошибок: {stats['failed']}")
        lines.append(f"Пропущено (невалидные): {stats['skipped']}")
        lines.append("="*80)
        
        if stats['success'] > 0:
            lines.append("\nУСПЕШНО СКАЧАННЫЕ ВИДЕО:")
            for result in stats['results']:
                if result.get('success'):
                    size_mb = result['file_size'] / (1024 * 1024)
                    duration_min = result['duration'] / 60
                    lines.append(f"  ✓ [{result['index']}] {result['author']} - {result['title']}")
                    lines.append(f"     Размер: {size_mb:.1f} MB, Длительность: {duration_min:.1f} мин")
                    lines.append(f"     Файл: {result['filename']}")
                    lines.append("")
        
        if stats['failed'] > 0:
            lines.append("\nОШИБКИ:")
            for result in stats['results']:
                if not result.get('success'):
                    lines.append(f"  ✗ [{result['index']}] {result['url']}")
                    lines.append(f"    Ошибка: {result.get('error', 'Unknown error')}")
                    lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Получает статус обработки по ID пакета"""
        report_path = os.path.join(self.download_dir, f"report_{batch_id}.json")
        if os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None


# Создаем экземпляр для использования в других модулях
default_batch_video_downloader = None

def get_batch_video_downloader():
    """Получить или создать экземпляр пакетного загрузчика видео"""
    global default_batch_video_downloader
    if default_batch_video_downloader is None:
        default_batch_video_downloader = BatchVideoDownloader()
    return default_batch_video_downloader


# CLI интерфейс
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Пакетное скачивание видео из TikTok")
    parser.add_argument("urls_file", help="Файл со ссылками (по одной на строку)")
    parser.add_argument("-q", "--quality", choices=['high', 'low'], default='high',
                       help="Качество видео")
    parser.add_argument("-w", "--workers", type=int, default=3,
                       help="Количество параллельных загрузок")
    parser.add_argument("-o", "--output", help="Имя ZIP архива")
    
    args = parser.parse_args()
    
    try:
        # Читаем ссылки из файла
        with open(args.urls_file, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"Загружено {len(urls)} ссылок из {args.urls_file}")
        
        # Скачиваем
        downloader = BatchVideoDownloader()
        
        async def run():
            zip_path, stats = await downloader.download_batch(
                urls,
                quality=args.quality,
                max_workers=args.workers,
                zip_name=args.output
            )
            
            print(f"\n✅ Готово! Результат сохранен в: {zip_path}")
            print(f"Успешно: {stats['success']}, Ошибок: {stats['failed']}")
        
        asyncio.run(run())
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        exit(1)
