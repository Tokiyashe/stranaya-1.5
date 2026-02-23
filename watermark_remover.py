"""
Gemini Watermark Remover - Исправленная версия с inpainting
Использует метод заполнения области (inpainting) вместо обратного альфа-смешивания
"""

import os
import numpy as np
from PIL import Image, ImageDraw
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
    Использует метод inpainting (заполнение области)
    """
    
    def __init__(self, mask_dir: Optional[str] = None):
        """
        Инициализация
        """
        self.mask_dir = mask_dir
        logger.info("GeminiWatermarkRemover инициализирован")
    
    def detect_watermark_region(self, image: Image.Image) -> Tuple[int, int, int, int]:
        """
        Определяет область с водяным знаком
        
        Returns:
            (x, y, width, height) координаты области с водяным знаком
        """
        width, height = image.size
        
        # Gemini ставит водяной знак в правый нижний угол
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
    
    def remove_watermark_inpaint(self, 
                                image: Image.Image,
                                region: Tuple[int, int, int, int]) -> Image.Image:
        """
        Удаляет водяной знак методом inpainting (заполнение области)
        """
        x, y, w, h = region
        
        # Создаем копию изображения
        result = image.copy()
        
        # Получаем область вокруг водяного знака для выборки цветов
        # Берем пиксели сверху и слева от области
        sample_top = max(0, y - 10)
        sample_left = max(0, x - 10)
        sample_right = min(image.width, x + w + 10)
        sample_bottom = min(image.height, y + h + 10)
        
        # Собираем образцы цветов из окружающей области
        samples = []
        
        # Область сверху
        if y > 10:
            top_region = image.crop((x, sample_top, x + w, y))
            samples.extend(list(top_region.getdata()))
        
        # Область слева
        if x > 10:
            left_region = image.crop((sample_left, y, x, y + h))
            samples.extend(list(left_region.getdata()))
        
        # Если нет образцов, используем средний цвет изображения
        if not samples:
            # Берем средний цвет всей картинки
            img_array = np.array(image)
            avg_color = tuple(np.mean(img_array, axis=(0, 1)).astype(int))
            fill_color = avg_color
        else:
            # Усредняем образцы
            avg_color = tuple(np.mean(samples, axis=0).astype(int))
            fill_color = avg_color
        
        # Рисуем прямоугольник с усредненным цветом
        draw = ImageDraw.Draw(result)
        draw.rectangle([x, y, x + w, y + h], fill=fill_color)
        
        # Добавляем небольшое размытие для естественности
        from PIL import ImageFilter
        # Вырезаем область, размываем и вставляем обратно
        region_img = result.crop((x, y, x + w, y + h))
        blurred = region_img.filter(ImageFilter.GaussianBlur(radius=2))
        result.paste(blurred, (x, y))
        
        return result
    
    def remove_watermark_advanced(self,
                                 image: Image.Image,
                                 region: Tuple[int, int, int, int]) -> Image.Image:
        """
        Продвинутое удаление с интерполяцией
        """
        x, y, w, h = region
        result = image.copy()
        
        # Конвертируем в numpy для обработки
        img_array = np.array(result)
        
        # Создаем маску области водяного знака
        mask = np.zeros(img_array.shape[:2], dtype=bool)
        mask[y:y+h, x:x+w] = True
        
        # Для каждого пикселя в маске
        for i in range(y, y + h):
            for j in range(x, x + w):
                # Берем среднее значение соседних пикселей
                neighbors = []
                
                # Соседи сверху
                if i > 0:
                    neighbors.append(img_array[i-1, j])
                # Снизу
                if i < img_array.shape[0] - 1:
                    neighbors.append(img_array[i+1, j])
                # Слева
                if j > 0:
                    neighbors.append(img_array[i, j-1])
                # Справа
                if j < img_array.shape[1] - 1:
                    neighbors.append(img_array[i, j+1])
                
                if neighbors:
                    # Усредняем соседей
                    avg_color = np.mean(neighbors, axis=0).astype(np.uint8)
                    img_array[i, j] = avg_color
        
        # Конвертируем обратно в изображение
        result = Image.fromarray(img_array)
        
        return result
    
    def remove_watermark(self, 
                        image: Union[str, Image.Image, np.ndarray, bytes],
                        output_path: Optional[str] = None,
                        force_size: Optional[str] = None,
                        method: str = "inpaint") -> Union[Image.Image, bytes, str]:
        """
        Удаляет водяной знак Gemini с изображения
        
        Args:
            image: путь к файлу, PIL Image, numpy array или байты
            output_path: если указан, сохраняет результат
            force_size: принудительный размер ('small' или 'large')
            method: метод удаления ('inpaint' или 'advanced')
        """
        try:
            # Загрузка изображения
            input_is_bytes = False
            
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Файл не найден: {image}")
                img = Image.open(image).convert('RGB')
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
            
            # Определяем размер водяного знака
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
            
            logger.info(f"Размер изображения: {width}x{height}")
            
            # Координаты области с водяным знаком
            x = width - margin - logo_size
            y = height - margin - logo_size
            
            # Проверяем, что область в пределах изображения
            if x < 0 or y < 0 or x + logo_size > width or y + logo_size > height:
                logger.warning("Область водяного знака вне изображения, возможно знака нет")
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
            
            region = (x, y, logo_size, logo_size)
            logger.info(f"Область водяного знака: x={x}, y={y}, размер={logo_size}")
            
            # Выбираем метод удаления
            if method == "advanced":
                result = self.remove_watermark_advanced(img, region)
            else:
                result = self.remove_watermark_inpaint(img, region)
            
            # Сохранение если нужно
            if output_path:
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


# Создаем экземпляр для использования в других модулях
default_remover = None

def get_remover(mask_dir: Optional[str] = None):
    """Получить или создать экземпляр удалителя водяных знаков"""
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
    parser.add_argument("-m", "--method", choices=['inpaint', 'advanced'], default='inpaint')
    
    args = parser.parse_args()
    
    try:
        remover = GeminiWatermarkRemover()
        output = args.output or args.input.replace('.', '_clean.')
        remover.remove_watermark(
            args.input, 
            output, 
            force_size=args.size,
            method=args.method
        )
        print(f"✅ Готово: {output}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        exit(1)
