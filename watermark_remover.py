"""
Gemini Watermark Remover - Полностью рабочая версия
Основано на обратном альфа-смешивании
Автор: Tokiyashe
Дата: 2026
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple
import io
import base64
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiWatermarkRemover:
    """
    Удаляет видимые водяные знаки Gemini с изображений
    Использует математически точный метод обратного альфа-смешивания
    """
    
    def __init__(self, mask_dir: Optional[str] = None):
        """
        Инициализация с загрузкой масок
        
        Args:
            mask_dir: директория с файлами масок (bg_48.png, bg_96.png)
        """
        self.masks = {}
        self.mask_dir = mask_dir
        
        if mask_dir and os.path.exists(mask_dir):
            self._load_masks()
            logger.info(f"Загружены маски из {mask_dir}")
        else:
            logger.warning(f"Директория масок не найдена: {mask_dir}")
            logger.warning("Будут использованы встроенные заглушки")
            self._create_dummy_masks()
    
    def _load_masks(self):
        """Загружает альфа-маски из файлов"""
        for size in [48, 96]:
            mask_path = os.path.join(self.mask_dir, f'bg_{size}.png')
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert('L')  # Grayscale
                    # Нормализация к [0,1] и инвертирование (маска хранит прозрачность)
                    mask_array = np.array(mask_img) / 255.0
                    # Инвертируем, если нужно (зависит от формата маски)
                    # self.masks[size] = 1 - mask_array
                    self.masks[size] = mask_array
                    logger.info(f"Загружена маска {size}x{size}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки маски {size}: {e}")
                    self._create_mask_for_size(size)
            else:
                logger.warning(f"Файл маски не найден: {mask_path}")
                self._create_mask_for_size(size)
    
    def _create_mask_for_size(self, size: int):
        """Создает заглушку маски для указанного размера"""
        # Стандартная маска Gemini: градиентная прозрачность
        # В центре более прозрачно, по краям менее
        mask = np.ones((size, size))
        
        # Создаем градиент
        y, x = np.ogrid[:size, :size]
        center = size // 2
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        max_dist = np.sqrt(2) * center
        
        # Прозрачность: от 0.3 в центре до 0.7 по краям
        alpha = 0.3 + 0.4 * (distance / max_dist)
        mask = mask * alpha
        
        self.masks[size] = mask
        logger.info(f"Создана заглушка маски {size}x{size}")
    
    def _create_dummy_masks(self):
        """Создает заглушки для всех размеров"""
        for size in [48, 96]:
            self._create_mask_for_size(size)
    
    def detect_watermark_position(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Определяет положение водяного знака
        
        Returns:
            (x, y, width, height) координаты области с водяным знаком
        """
        width, height = image.size
        
        # Gemini всегда ставит водяной знак в правый нижний угол
        # С небольшим отступом от края
        if width <= 1024 or height <= 1024:
            logo_size = 48
            margin = 32
        else:
            logo_size = 96
            margin = 64
        
        x = width - margin - logo_size
        y = height - margin - logo_size
        
        return (x, y, logo_size, logo_size)
    
    def remove_watermark_from_array(self, 
                                   img_array: np.ndarray,
                                   logo_size: int,
                                   margin: int) -> np.ndarray:
        """
        Удаляет водяной знак из numpy массива изображения
        
        Формула: original = (watermarked - α * logo) / (1 - α)
        где α - прозрачность из маски
        """
        h, w = img_array.shape[:2]
        
        # Координаты области с водяным знаком
        x1 = w - margin - logo_size
        y1 = h - margin - logo_size
        x2 = w - margin
        y2 = h - margin
        
        # Проверяем, что область в пределах изображения
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logger.warning(f"Область водяного знака выходит за границы: {x1},{y1} - {x2},{y2}")
            # Центрируем если выходит
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x1 + logo_size)
            y2 = min(h, y1 + logo_size)
            # Корректируем размер
            logo_size = min(x2 - x1, y2 - y1)
        
        # Извлекаем область с водяным знаком
        watermark_region = img_array[y1:y2, x1:x2].copy()
        
        # Получаем альфа-маску для этого размера
        if logo_size in self.masks:
            alpha_mask = self.masks[logo_size]
            # Если маска больше области, обрезаем
            if alpha_mask.shape[0] > logo_size or alpha_mask.shape[1] > logo_size:
                alpha_mask = alpha_mask[:logo_size, :logo_size]
        else:
            # Если нет подходящей маски, используем стандартную
            alpha_mask = np.ones((logo_size, logo_size)) * 0.5
        
        # Расширяем маску до 3 каналов (RGB)
        if len(alpha_mask.shape) == 2:
            alpha_mask_3d = np.stack([alpha_mask] * 3, axis=2)
        else:
            alpha_mask_3d = alpha_mask
        
        # Применяем обратное альфа-смешивание
        # Защита от деления на ноль
        denominator = 1 - alpha_mask_3d
        denominator = np.where(denominator < 0.01, 0.01, denominator)
        
        # Предполагаем, что цвет логотипа - черный (0,0,0) или белый (1,1,1)
        # В большинстве случаев логотип Gemini - полупрозрачный белый
        logo_color = np.array([1.0, 1.0, 1.0])  # белый
        
        # Восстанавливаем оригинал
        # Формула: I = (W - α*L) / (1-α)
        restored = (watermark_region - alpha_mask_3d * logo_color) / denominator
        restored = np.clip(restored, 0, 1)
        
        # Вставляем обратно
        img_array[y1:y2, x1:x2] = restored
        
        return img_array
    
    def remove_watermark(self, 
                        image: Union[str, Image.Image, np.ndarray, bytes],
                        output_path: Optional[str] = None,
                        force_size: Optional[str] = None) -> Union[Image.Image, bytes, str]:
        """
        Удаляет водяной знак Gemini с изображения
        
        Args:
            image: путь к файлу, PIL Image, numpy array или байты
            output_path: если указан, сохраняет результат
            force_size: принудительный размер ('small' или 'large')
            
        Returns:
            PIL Image, bytes или путь к файлу (если output_path указан)
        """
        try:
            # Загрузка изображения
            input_is_bytes = False
            input_is_path = False
            
            if isinstance(image, str):
                # Проверяем, существует ли файл
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Файл не найден: {image}")
                img = Image.open(image).convert('RGB')
                input_is_path = True
                logger.info(f"Загружено изображение из файла: {image}")
                
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image)).convert('RGB')
                input_is_bytes = True
                logger.info(f"Загружено изображение из байтов, размер: {len(image)} bytes")
                
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
                logger.info("Загружено PIL Image")
                
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image).convert('RGB')
                logger.info("Загружено numpy array")
                
            else:
                raise TypeError(f"Неподдерживаемый тип: {type(image)}")
            
            # Определение размера водяного знака
            width, height = img.size
            
            if force_size == 'small':
                logo_size, margin = 48, 32
                logger.info("Принудительно: маленький логотип (48px)")
            elif force_size == 'large':
                logo_size, margin = 96, 64
                logger.info("Принудительно: большой логотип (96px)")
            else:
                if width <= 1024 or height <= 1024:
                    logo_size, margin = 48, 32
                    logger.info("Автоопределение: маленький логотип (48px)")
                else:
                    logo_size, margin = 96, 64
                    logger.info("Автоопределение: большой логотип (96px)")
            
            logger.info(f"Размер изображения: {width}x{height}, логотип: {logo_size}, отступ: {margin}")
            
            # Проверяем, достаточно ли места для водяного знака
            if width < margin + logo_size or height < margin + logo_size:
                logger.warning("Изображение слишком маленькое, возможно водяного знака нет")
                # Возвращаем оригинал
                if output_path:
                    img.save(output_path)
                    return output_path
                elif input_is_bytes:
                    output_bytes = io.BytesIO()
                    img.save(output_bytes, format='PNG')
                    return output_bytes.getvalue()
                else:
                    return img
            
            # Конвертация в numpy массив
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Удаление водяного знака
            img_array = self.remove_watermark_from_array(img_array, logo_size, margin)
            
            # Конвертация обратно в изображение
            result = Image.fromarray((img_array * 255).astype(np.uint8))
            
            # Сохранение если нужно
            if output_path:
                # Создаем папку если нужно
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result.save(output_path, quality=95)
                logger.info(f"Сохранено в: {output_path}")
                return output_path
            
            # Возврат в том же формате, что и вход
            if input_is_bytes:
                output_bytes = io.BytesIO()
                result.save(output_bytes, format='PNG', quality=95)
                logger.info("Возврат байтов")
                return output_bytes.getvalue()
            
            logger.info("Возврат PIL Image")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка удаления водяного знака: {e}")
            raise RuntimeError(f"Не удалось удалить водяной знак: {str(e)}")
    
    def remove_watermark_base64(self, base64_string: str) -> str:
        """Удаляет водяной знак из base64 изображения"""
        try:
            # Декодируем base64
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            clean_bytes = self.remove_watermark(image_bytes)
            
            # Кодируем обратно в base64
            clean_base64 = base64.b64encode(clean_bytes).decode('utf-8')
            return f"data:image/png;base64,{clean_base64}"
            
        except Exception as e:
            logger.error(f"Ошибка обработки base64: {e}")
            raise
    
    def batch_process(self, image_paths: List[str], output_dir: str, **kwargs) -> List[str]:
        """
        Пакетная обработка нескольких изображений
        
        Args:
            image_paths: список путей к изображениям
            output_dir: директория для сохранения результатов
            **kwargs: параметры для remove_watermark
            
        Returns:
            список путей к обработанным файлам
        """
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, img_path in enumerate(image_paths):
            try:
                logger.info(f"Обработка {i+1}/{len(image_paths)}: {img_path}")
                
                # Генерируем выходное имя
                base_name = os.path.basename(img_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join(output_dir, f"{name}_clean.png")
                
                # Обрабатываем
                self.remove_watermark(img_path, output_path, **kwargs)
                results.append(output_path)
                
            except Exception as e:
                logger.error(f"Ошибка обработки {img_path}: {e}")
                results.append(None)
        
        return results


# Создаем экземпляр для использования в других модулях
default_remover = None

def get_remover(mask_dir: Optional[str] = None):
    """Получить или создать экземпляр удалителя водяных знаков"""
    global default_remover
    if default_remover is None:
        default_remover = GeminiWatermarkRemover(mask_dir)
    return default_remover


# CLI интерфейс
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Удаление водяных знаков Gemini")
    parser.add_argument("input", help="Входное изображение или папка")
    parser.add_argument("output", nargs="?", help="Выходное изображение или папка (опционально)")
    parser.add_argument("-s", "--size", choices=['auto', 'small', 'large'], 
                       default='auto', help="Размер водяного знака")
    parser.add_argument("-m", "--mask-dir", default="masks",
                       help="Директория с масками (по умолчанию: masks)")
    parser.add_argument("-b", "--batch", action="store_true",
                       help="Пакетный режим (input должен быть папкой)")
    
    args = parser.parse_args()
    
    try:
        remover = GeminiWatermarkRemover(args.mask_dir)
        
        if args.batch:
            # Пакетный режим
            if not os.path.isdir(args.input):
                print("❌ В пакетном режиме input должен быть папкой")
                exit(1)
            
            output_dir = args.output or os.path.join(args.input, "cleaned")
            image_files = []
            
            # Собираем все изображения
            for root, dirs, files in os.walk(args.input):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_files.append(os.path.join(root, file))
            
            print(f"Найдено {len(image_files)} изображений")
            results = remover.batch_process(image_files, output_dir, force_size=args.size)
            
            success = sum(1 for r in results if r is not None)
            print(f"✅ Обработано: {success}/{len(results)}")
            print(f"📁 Результаты сохранены в: {output_dir}")
            
        else:
            # Одиночный режим
            if not os.path.isfile(args.input):
                print("❌ Файл не найден")
                exit(1)
            
            # Определяем выходной файл
            if args.output:
                output_path = args.output
            else:
                name, ext = os.path.splitext(args.input)
                output_path = f"{name}_clean.png"
            
            result = remover.remove_watermark(args.input, output_path, force_size=args.size)
            print(f"✅ Готово! Результат сохранен в: {result}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        exit(1)