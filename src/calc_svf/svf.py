import cv2
import numpy as np
import math
import logging

def calculate_svf(fisheye_path, svf_rings):
    """
    计算天空视角因子(SVF)

    :param fisheye_path: 鱼眼图像路径
    :param svf_rings: 计算SVF的环数
    :return: SVF值
    """
    try:
        img = cv2.imread(fisheye_path)
        if img is None:
            raise Exception(f"无法读取鱼眼图像: {fisheye_path}")

        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 使用阈值分割天空（假设天空像素值较高）
        _, sky_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        h, w = sky_mask.shape
        center_x, center_y = w // 2, h // 2
        radius = min(center_x, center_y)
        ring_width = radius // svf_rings

        svf = 0.0
        for i in range(1, svf_rings + 1):
            inner_r, outer_r = (i - 1) * ring_width, i * ring_width

            # 创建环形掩码
            mask = np.zeros_like(sky_mask)
            cv2.circle(mask, (center_x, center_y), outer_r, 255, -1)
            cv2.circle(mask, (center_x, center_y), inner_r, 0, -1)

            # 计算环形区域内的天空像素比例
            sky_pixels = cv2.countNonZero(cv2.bitwise_and(sky_mask, mask))
            total_pixels = cv2.countNonZero(mask)

            if total_pixels == 0:
                continue

            # 计算该环对应的天顶角
            angle = math.pi * (i - 0.5) / (2 * svf_rings)

            # 累加SVF贡献
            svf += (1 / (2 * svf_rings)) * math.sin(angle) * math.cos(angle) * (sky_pixels / total_pixels) * 2 * math.pi

        return round(svf, 4)

    except Exception as e:
        logging.error(f"计算SVF失败: {str(e)}")
        raise