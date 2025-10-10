import os
import cv2
import numpy as np
from PIL import Image
import math
import logging
import uuid

def generate_fisheye_image(image_paths, output_path, temp_dir):
    """
    按 0,90,180,270 顺序水平拼接图像

    :param image_paths: 四个方向图片的字典 {direction: filepath}
    :param output_path: 输出拼接图像路径
    :param temp_dir: 临时目录
    :return: 是否成功
    """
    try:
        required_directions = ['0', '90', '180', '270']
        if not all(direction in image_paths for direction in required_directions):
            missing = [d for d in required_directions if d not in image_paths]
            logging.error(f"缺少方向图片: {missing}")
            return False

        # 读取并对齐高度后水平拼接
        pil_images = [Image.open(image_paths[d]).convert('RGB') for d in required_directions]
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pano_tmp = os.path.join(temp_dir, f"pano_{uuid.uuid4().hex[:8]}.png")
        canvas.save(pano_tmp)

        img = cv2.imread(pano_tmp)
        if img is None:
            logging.error("无法读取临时拼接图像")
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
        cv2.imwrite(output_path, new_img)
        logging.info(f"鱼眼图已保存到: {output_path}")
        return True
    except Exception as e:
        logging.error(f"拼接图生成失败: {str(e)}")
        return False