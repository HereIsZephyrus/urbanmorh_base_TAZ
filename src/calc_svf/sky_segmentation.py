from PIL import Image
from transformers import pipeline
import os
import tempfile
from io import BytesIO

# 初始化分割模型
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")


def extract_sky_mask_from_image(image):
    """
    从 PIL.Image 或 bytes 中提取天空掩码，返回 PIL.Image 或 None。
    该函数用于内存处理，避免不必要的磁盘 IO。
    """
    if not isinstance(image, Image.Image):
        # assume bytes
        image = Image.open(BytesIO(image))
    results = segmenter(image)
    sky_mask = None
    for result in results:
        if result.get('label') == 'sky':
            sky_mask = result.get('mask')  # 这是一个PIL图像
            break
    return sky_mask


def extract_and_save_sky_mask(image_path, output_path):
    """
    兼容旧接口：从磁盘图片提取天空掩码并保存到磁盘（保留，但原程序现在可选择不使用此函数）。
    """
    image = Image.open(image_path)
    sky_mask = extract_sky_mask_from_image(image)
    if sky_mask is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sky_mask.save(output_path)
        print(f"天空掩码已保存: {output_path}")
    else:
        print(f"未在图片中找到天空: {image_path}")

def group_images_by_coord(mask_dir):
    groups = {}
    for file in os.listdir(mask_dir):
        if file.endswith('.png') and file.startswith('mask_'):
            parts = file.replace('mask_', '').replace('.png', '').split('_')
            if len(parts) == 4:
                wgs_x, wgs_y, heading, pitch = parts
                coord_key = f"{wgs_x}_{wgs_y}"
                if coord_key not in groups:
                    groups[coord_key] = {}
                groups[coord_key][heading] = os.path.join(mask_dir, file)
    return groups

if __name__ == "__main__":
    input_dir = "pictures"
    output_dir = "sky_masks"
    fisheye_dir = "fisheye_output"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fisheye_dir, exist_ok=True)
    temp_dir = tempfile.gettempdir()
    # 1. 先批量提取天空掩码
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.png'):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, f"mask_{file_name}")
            extract_and_save_sky_mask(input_path, output_path)
    # 2. 合成鱼眼图
    groups = group_images_by_coord(output_dir)
    for coord_key, image_paths in groups.items():
        if all(h in image_paths for h in ['0', '90', '180', '270']):
            output_path = os.path.join(fisheye_dir, f'fisheye_{coord_key}.png')
            success = generate_fisheye_image(image_paths, output_path, temp_dir)
            if success:
                print(f"合成鱼眼图成功: {output_path}")
            else:
                print(f"合成鱼眼图失败: {output_path}")
        else:
            print(f"坐标 {coord_key} 图片不全，跳过")