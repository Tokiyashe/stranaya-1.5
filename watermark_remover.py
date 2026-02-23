"""
Gemini Watermark Remover - Исправленная версия
Правильное обратное альфа-смешивание
"""

import os
import numpy as np
from PIL import Image
from typing import Optional, Union, Tuple, List
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
        """
        self.masks = {}
        self.mask_dir = mask_dir
        
        if mask_dir and os.path.exists(mask_dir):
            self._load_masks()
        else:
            logger.warning(f"Директория масок не найдена: {mask_dir}")
            logger.warning("Будут использованы встроенные маски")
            self._create_dummy_masks()
    
    def _load_masks(self):
        """Загружает альфа-маски из файлов"""
        for size in [48, 96]:
            mask_path = os.path.join(self.mask_dir, f'bg_{size}.png')
            if os.path.exists(mask_path):
                try:
                    mask_img = Image.open(mask_path).convert('L')
                    mask_array = np.array(mask_img) / 255.0
                    # Инвертируем маску (более светлые пиксели = более прозрачные)
                    self.masks[size] = 1 - mask_array
                    logger.info(f"Загружена маска {size}x{size}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки маски {size}: {e}")
                    self._create_mask_for_size(size)
            else:
                logger.warning(f"Файл маски не найден: {mask_path}")
                self._create_mask_for_size(size)
    
    def _create_mask_for_size(self, size: int):
        """Создает заглушку маски для указанного размера"""
        # Стандартная маска Gemini: значение прозрачности 0.3-0.7
        mask = np.ones((size, size)) * 0.5  # базовая прозрачность 50%
        
        # Добавляем градиент от центра
        y, x = np.ogrid[:size, :size]
        center = size // 2
        distance = np.sqrt((x - center)**2 + (y - center)**2)
        max_dist = np.sqrt(2) * center
        
        # Прозрачность: от 0.7 в центре до 0.3 по краям
        gradient = 0.7 - 0.4 * (distance / max_dist)
        mask = mask * gradient
        
        self.masks[size] = mask
        logger.info(f"Создана маска-заглушка {size}x{size}")
    
    def remove_watermark_from_array(self, 
                                   img_array: np.ndarray,
                                   logo_size: int,
                                   margin: int) -> np.ndarray:
        """
        Удаляет водяной знак из numpy массива изображения
        
        Правильная формула: 
        original = (watermarked - alpha * logo_color) / (1 - alpha)
        где alpha - прозрачность водяного знака
        logo_color - цвет водяного знака (обычно белый [1,1,1])
        """
        h, w = img_array.shape[:2]
        
        # Координаты области с водяным знаком
        x1 = w - margin - logo_size
        y1 = h - margin - logo_size
        x2 = w - margin
        y2 = h - margin
        
        # Проверяем границы
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            logger.warning("Область водяного знака вне изображения")
            return img_array
        
        # Извлекаем область с водяным знаком
        watermark_region = img_array[y1:y2, x1:x2].copy()
        
        # Получаем альфа-маску
        if logo_size in self.masks:
            alpha = self.masks[logo_size]
            # Обрезаем если маска больше
            if alpha.shape[0] > logo_size or alpha.shape[1] > logo_size:
                alpha = alpha[:logo_size, :logo_size]
        else:
            alpha = np.ones((logo_size, logo_size)) * 0.5
        
        # Расширяем маску до 3 каналов
        alpha_3d = np.stack([alpha] * 3, axis=2)
        
        # Цвет водяного знака (обычно белый)
        logo_color = np.array([1.0, 1.0, 1.0])
        
        # Защита от деления на ноль
        alpha_3d = np.clip(alpha_3d, 0.01, 0.99)
        
        # Восстанавливаем оригинал
        # Формула: original = (watermarked - alpha * logo_color) / (1 - alpha)
        restored = (watermark_region - alpha_3d * logo_color) / (1 - alpha_3d)
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
        """
        try:
            # Загрузка изображения
            input_is_bytes = False
            
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Файл не найден: {image}")
                img = Image.open(image).convert('RGB')
                
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image)).convert('RGB')
                input_is_bytes = True
                
            elif isinstance(image, Image.Image):
                img = image.convert('RGB')
                
            elif isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                img = Image.fromarray(image).convert('RGB')
                
            else:
                raise TypeError(f"Неподдерживаемый тип: {type(image)}")
            
            # Определяем размер водяного знака
            width, height = img.size
            
            if force_size == 'small':
                logo_size, margin = 48, 32
            elif force_size == 'large':
                logo_size, margin = 96, 64
            else:
                if width <= 1024 or height <= 1024:
                    logo_size, margin = 48, 32
                else:
                    logo_size, margin = 96, 64
            
            logger.info(f"Изображение: {width}x{height}, логотип: {logo_size}, отступ: {margin}")
            
            # Конвертация в numpy
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Удаляем водяной знак
            img_array = self.remove_watermark_from_array(img_array, logo_size, margin)
            
            # Конвертация обратно
            result = Image.fromarray((img_array * 255).astype(np.uint8))
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                result.save(output_path, quality=95)
                return output_path
            
            if input_is_bytes:
                output_bytes = io.BytesIO()
                result.save(output_bytes, format='PNG')
                return output_bytes.getvalue()
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка: {e}")
            raise


# Создаем экземпляр
default_remover = None

def get_remover(mask_dir: Optional[str] = None):
    global default_remover
    if default_remover is None:
        default_remover = GeminiWatermarkRemover(mask_dir)
    return default_remover


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Удаление водяных знаков Gemini")
    parser.add_argument("input", help="Входное изображение")
    parser.add_argument("output", nargs="?", help="Выходное изображение")
    parser.add_argument("-s", "--size", choices=['auto', 'small', 'large'], default='auto')
    
    args = parser.parse_args()
    
    remover = GeminiWatermarkRemover("masks/")
    output = args.output or args.input.replace('.', '_clean.')
    remover.remove_watermark(args.input, output, force_size=args.size)
    print(f"✅ Готово: {output}")
