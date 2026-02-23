"""
Batch Watermark Remover - Пакетное удаление водяных знаков из ZIP архива
Автор: Tokiyashe
Дата: 2026
"""

import os
import zipfile
import shutil
import json
from typing import List, Tuple, Optional, Dict
from PIL import Image
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path

from watermark_remover import get_remover

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Поддерживаемые форматы изображений
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.jfif'}

class BatchWatermarkRemover:
    """
    Пакетное удаление водяных знаков из ZIP архива
    """
    
    def __init__(self, upload_dir: str = "uploads", output_dir: str = "downloads/batch"):
        """
        Инициализация обработчика
        
        Args:
            upload_dir: директория для временных загрузок
            output_dir: директория для результатов
        """
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        self.watermark_remover = get_remover(mask_dir="masks/")
        
        # Создаем необходимые папки
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"BatchWatermarkRemover инициализирован")
        logger.info(f"  Upload dir: {upload_dir}")
        logger.info(f"  Output dir: {output_dir}")
    
    def _safe_filename(self, filename: str) -> str:
        """Очищает имя файла от опасных символов"""
        import re
        # Заменяем потенциально опасные символы
        filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
        # Убираем path traversal
        filename = filename.replace('..', '_')
        # Ограничиваем длину
        if len(filename) > 200:
            name, ext = os.path.splitext(filename)
            filename = name[:190] + ext
        return filename
    
    def extract_zip(self, zip_path: str) -> Tuple[str, List[str], List[str], Dict]:
        """
        Распаковывает ZIP и возвращает путь к папке и списки файлов
        
        Returns:
            (extract_path, all_files, image_files, structure_info)
        """
        # Создаем уникальную папку для распаковки
        extract_id = str(uuid.uuid4())[:8]
        extract_path = os.path.join(self.upload_dir, f"extract_{extract_id}")
        os.makedirs(extract_path, exist_ok=True)
        
        all_files = []
        image_files = []
        structure_info = {
            'folders': set(),
            'files': []
        }
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Получаем список всех файлов
                file_list = zip_ref.namelist()
                
                for file_name in file_list:
                    # Запоминаем структуру папок
                    if '/' in file_name:
                        folder = file_name.rsplit('/', 1)[0]
                        structure_info['folders'].add(folder)
                    
                    # Пропускаем папки
                    if file_name.endswith('/'):
                        continue
                    
                    # Очищаем имя файла
                    safe_name = self._safe_filename(file_name)
                    
                    # Создаем полный путь для распаковки
                    target_path = os.path.join(extract_path, safe_name)
                    
                    # Создаем подпапки если нужно
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    
                    # Извлекаем файл
                    zip_ref.extract(file_name, extract_path)
                    
                    # Переименовываем если нужно (для безопасности)
                    extracted_file = os.path.join(extract_path, file_name)
                    if extracted_file != target_path and os.path.exists(extracted_file):
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        shutil.move(extracted_file, target_path)
                        
                        # Удаляем пустые папки
                        self._remove_empty_folders(os.path.dirname(extracted_file))
                    
                    all_files.append(target_path)
                    
                    # Проверяем, является ли файл изображением
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in SUPPORTED_FORMATS:
                        image_files.append(target_path)
                        structure_info['files'].append({
                            'path': target_path,
                            'original_name': file_name,
                            'size': os.path.getsize(target_path)
                        })
            
            structure_info['folders'] = list(structure_info['folders'])
            logger.info(f"Распаковано {len(all_files)} файлов, из них {len(image_files)} изображений")
            
            return extract_path, all_files, image_files, structure_info
            
        except Exception as e:
            logger.error(f"Ошибка распаковки ZIP: {e}")
            # Очищаем при ошибке
            shutil.rmtree(extract_path, ignore_errors=True)
            raise
    
    def _remove_empty_folders(self, path: str):
        """Удаляет пустые папки рекурсивно"""
        if not os.path.isdir(path):
            return
        
        # Удаляем пустые подпапки
        for root, dirs, files in os.walk(path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                except:
                    pass
        
        # Удаляем саму папку если пуста
        try:
            if not os.listdir(path):
                os.rmdir(path)
        except:
            pass
    
    def process_single_image(self, 
                            input_path: str, 
                            output_dir: str,
                            size: str = "auto",
                            original_name: str = None) -> Dict:
        """
        Обрабатывает одно изображение
        
        Returns:
            словарь с информацией о результате
        """
        try:
            # Определяем имя файла
            if original_name is None:
                original_name = os.path.basename(input_path)
            
            name, ext = os.path.splitext(original_name)
            # Очищаем имя от лишних символов
            name = self._safe_filename(name)
            
            # Генерируем выходное имя
            output_filename = f"{name}_clean.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Удаляем водяной знак
            start_time = time.time()
            
            force_size = None if size == "auto" else size
            self.watermark_remover.remove_watermark(
                input_path,
                output_path,
                force_size=force_size
            )
            
            process_time = time.time() - start_time
            
            return {
                'original': original_name,
                'output': output_filename,
                'output_path': output_path,
                'size': os.path.getsize(output_path),
                'time': round(process_time, 2),
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Ошибка обработки {input_path}: {e}")
            return {
                'original': os.path.basename(input_path),
                'success': False,
                'error': str(e)
            }
    
    def process_batch(self, 
                     zip_path: str,
                     size: str = "auto",
                     max_workers: int = 4,
                     keep_structure: bool = True) -> Tuple[str, Dict]:
        """
        Обрабатывает все изображения из ZIP архива
        
        Args:
            zip_path: путь к ZIP файлу
            size: размер водяного знака ('auto', 'small', 'large')
            max_workers: количество параллельных потоков
            keep_structure: сохранять структуру папок
            
        Returns:
            (output_zip_path, stats)
        """
        # Создаем временные папки
        batch_id = str(uuid.uuid4())[:8]
        temp_dir = os.path.join(self.upload_dir, f"batch_{batch_id}")
        output_dir = os.path.join(temp_dir, "processed")
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            'batch_id': batch_id,
            'total_files': 0,
            'total_images': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0,
            'results': [],
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': None,
            'total_time': None
        }
        
        try:
            start_total = time.time()
            
            # Распаковываем архив
            extract_path, all_files, image_files, structure = self.extract_zip(zip_path)
            stats['total_files'] = len(all_files)
            stats['total_images'] = len(image_files)
            stats['skipped'] = len(all_files) - len(image_files)
            
            logger.info(f"Начинаем обработку {len(image_files)} изображений")
            
            # Параллельная обработка
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                for img_path in image_files:
                    # Находим оригинальное имя для сохранения структуры
                    original_name = None
                    for file_info in structure['files']:
                        if file_info['path'] == img_path:
                            original_name = file_info['original_name']
                            break
                    
                    future = executor.submit(
                        self.process_single_image,
                        img_path,
                        output_dir,
                        size,
                        original_name
                    )
                    futures.append(future)
                
                # Собираем результаты
                for future in as_completed(futures):
                    result = future.result()
                    stats['results'].append(result)
                    
                    if result and result.get('success'):
                        stats['processed'] += 1
                    else:
                        stats['failed'] += 1
            
            # Создаем итоговый ZIP архив
            output_zip = os.path.join(self.output_dir, f"processed_{batch_id}.zip")
            
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Добавляем обработанные изображения
                if keep_structure:
                    # Пытаемся сохранить оригинальную структуру
                    for result in stats['results']:
                        if result.get('success'):
                            # Ищем оригинальный путь
                            original_path = None
                            for file_info in structure['files']:
                                if file_info['original_name'] == result['original']:
                                    original_path = file_info['original_name']
                                    break
                            
                            if original_path:
                                # Сохраняем структуру папок
                                zipf.write(result['output_path'], original_path)
                            else:
                                # Если не нашли, кладем в корень
                                zipf.write(result['output_path'], result['output'])
                else:
                    # Все в корень
                    for result in stats['results']:
                        if result.get('success'):
                            zipf.write(result['output_path'], result['output'])
                
                # Добавляем отчет
                stats['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                stats['total_time'] = round(time.time() - start_total, 2)
                
                report = self._generate_report(stats)
                zipf.writestr("report.txt", report)
                
                # Добавляем JSON отчет для машинной обработки
                json_report = json.dumps(stats, indent=2, default=str)
                zipf.writestr("report.json", json_report)
            
            logger.info(f"Готово: {output_zip}")
            logger.info(f"Обработано: {stats['processed']}/{stats['total_images']}")
            
            return output_zip, stats
            
        except Exception as e:
            logger.error(f"Ошибка пакетной обработки: {e}")
            raise
        finally:
            # Очищаем временные файлы
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                if 'extract_path' in locals():
                    shutil.rmtree(extract_path, ignore_errors=True)
            except:
                pass
    
    def _generate_report(self, stats: Dict) -> str:
        """Генерирует текстовый отчет"""
        lines = []
        lines.append("="*60)
        lines.append("ОТЧЕТ ОБ ОБРАБОТКЕ ИЗОБРАЖЕНИЙ")
        lines.append("="*60)
        lines.append(f"ID пакета: {stats['batch_id']}")
        lines.append(f"Начало: {stats['start_time']}")
        lines.append(f"Конец: {stats['end_time']}")
        lines.append(f"Общее время: {stats['total_time']} сек")
        lines.append("-"*60)
        lines.append(f"Всего файлов в архиве: {stats['total_files']}")
        lines.append(f"Изображений найдено: {stats['total_images']}")
        lines.append(f"Пропущено (не изображения): {stats['skipped']}")
        lines.append("-"*60)
        lines.append(f"Успешно обработано: {stats['processed']}")
        lines.append(f"Ошибок: {stats['failed']}")
        lines.append("="*60)
        
        if stats['failed'] > 0:
            lines.append("\nОШИБКИ:")
            for result in stats['results']:
                if not result.get('success'):
                    lines.append(f"  • {result['original']}: {result.get('error', 'Unknown error')}")
        
        lines.append("\nУСПЕШНО ОБРАБОТАННЫЕ ФАЙЛЫ:")
        for result in stats['results']:
            if result.get('success'):
                lines.append(f"  ✓ {result['original']} → {result['output']} ({result['time']} сек)")
        
        return "\n".join(lines)
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict]:
        """Получает статус обработки по ID пакета"""
        # Ищем файл отчета
        report_path = os.path.join(self.output_dir, f"report_{batch_id}.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                return json.load(f)
        return None


# Создаем экземпляр для использования в других модулях
default_batch_processor = None

def get_batch_processor():
    """Получить или создать экземпляр пакетного обработчика"""
    global default_batch_processor
    if default_batch_processor is None:
        default_batch_processor = BatchWatermarkRemover()
    return default_batch_processor


# CLI интерфейс
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Пакетное удаление водяных знаков")
    parser.add_argument("zip_file", help="ZIP файл с изображениями")
    parser.add_argument("-s", "--size", choices=['auto', 'small', 'large'], 
                       default='auto', help="Размер водяного знака")
    parser.add_argument("-w", "--workers", type=int, default=4,
                       help="Количество параллельных потоков")
    parser.add_argument("--no-structure", action='store_true',
                       help="Не сохранять структуру папок")
    
    args = parser.parse_args()
    
    try:
        processor = BatchWatermarkRemover()
        output_zip, stats = processor.process_batch(
            args.zip_file,
            size=args.size,
            max_workers=args.workers,
            keep_structure=not args.no_structure
        )
        
        print(f"\n✅ Готово! Результат сохранен в: {output_zip}")
        print(f"Обработано: {stats['processed']} из {stats['total_images']}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

        exit(1)
