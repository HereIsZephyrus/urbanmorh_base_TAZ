import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import math
import logging
import uuid

def _open_maybe(path_or_bytes_or_pil):
    """Helper to obtain a PIL.Image from a path, bytes or PIL.Image."""
    if isinstance(path_or_bytes_or_pil, Image.Image):
        return path_or_bytes_or_pil.convert('RGB')
    if isinstance(path_or_bytes_or_pil, (bytes, bytearray)):
        return Image.open(BytesIO(path_or_bytes_or_pil)).convert('RGB')
    # assume it's a path
    return Image.open(path_or_bytes_or_pil).convert('RGB')


def generate_fisheye_image(image_paths, output_path, temp_dir=None):
    """
    按 0,90,180,270 顺序水平拼接图像，支持输入为文件路径、bytes 或 PIL.Image 对象。

    :param image_paths: 四个方向图片的字典 {direction: filepath or bytes or PIL.Image}
    :param output_path: 输出拼接图像路径
    :param temp_dir: 兼容参数（不再需要）
    :return: 是否成功
    """
    try:
        required_directions = ['0', '90', '180', '270']
        if not all(direction in image_paths for direction in required_directions):
            missing = [d for d in required_directions if d not in image_paths]
            logging.error(f"缺少方向图片: {missing}")
            return False

        # 读取并对齐高度后水平拼接（支持多种输入类型）
        pil_images = [_open_maybe(image_paths[d]) for d in required_directions]
        heights = [im.height for im in pil_images]
        target_h = max(heights)
        resized = []
        for im in pil_images:
            if im.height != target_h:
                new_w = int(im.width * (target_h / im.height))
                im = im.resize((new_w, target_h), Image.BICUBIC)
            resized.append(im)
        total_w = sum(im.width for im in resized)
        canvas = Image.new('RGB', (total_w, target_h))
        x = 0
        for im in resized:
            canvas.paste(im, (x, 0))
            x += im.width

        # 直接用 numpy/cv2 处理，不再写入临时全景文件，减少磁盘 IO
        img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        if img is None:
            logging.error("无法构造拼接图像")
            return False
        rows, cols, c = img.shape
        R = int(cols / 2 / math.pi)
        D = R * 2
        cx = R
        cy = R
        new_img = np.zeros((D, D, c), dtype=np.uint8)
        for i in range(D):
            for j in range(D):
                r = math.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                if r > R:
                    continue
                tan_inv = math.atan((j - cy) / (i - cx + 1e-10))
                if i < cx:
                    theta = math.pi / 2 + tan_inv
                else:
                    theta = math.pi * 3 / 2 + tan_inv
                xp = int(math.floor(theta / (2 * math.pi) * cols))
                yp = int(math.floor(r / R * rows) - 1)
                xp = xp % cols
                if yp < 0:
                    yp = 0
                if yp >= rows:
                    yp = rows - 1
                new_img[j, i] = img[yp, xp]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, new_img)
        logging.info(f"鱼眼图已保存到: {output_path}")
        return True
    except Exception as e:
        logging.error(f"拼接图生成失败: {str(e)}")
        return False