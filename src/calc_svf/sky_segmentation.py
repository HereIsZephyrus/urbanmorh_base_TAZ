from PIL import Image
from transformers import pipeline
import os
import tempfile
from fisheye import generate_fisheye_image

# 初始化分割模型
segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")

def extract_and_save_sky_mask(image_path, output_path):
    image = Image.open(image_path)
    results = segmenter(image)
    sky_mask = None
    for result in results:
        if result['label'] == 'sky':
            sky_mask = result['mask']  # 这是一个PIL图像
            break
    if sky_mask is not None:
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