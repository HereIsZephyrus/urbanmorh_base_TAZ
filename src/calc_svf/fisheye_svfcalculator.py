import os
import asyncio
import logging
from retrying import retry
from fisheye import generate_fisheye_image
from svf import calculate_svf
import tempfile
from baiduStreetViewSpider import read_csv, getPanoId, grab_img_baidu, wgs2bd09mc
from PIL import Image, ImageDraw
from sky_segmentation import extract_and_save_sky_mask
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@retry(stop_max_attempt_number=3, wait_fixed=2000)
async def process_coordinate(coord_key, image_paths, output_dir, temp_dir, svf_rings):
    """
    处理单个坐标组的图片，生成鱼眼图并计算SVF。

    :param coord_key: 坐标键 (lng_lat)
    :param image_paths: 图片路径字典
    :param output_dir: 输出目录
    :param temp_dir: 临时目录
    :param svf_rings: SVF计算的环数
    :return: 处理结果
    """
    result = {
        "coordinate": coord_key,
        "success": False,
        "fisheye_path": None,
        "svf": None
    }

    try:
        logging.info(f"处理坐标组: {coord_key}")
        os.makedirs(output_dir, exist_ok=True)

        fisheye_filename = f"fisheye_{coord_key}.png"
        fisheye_path = os.path.join(output_dir, fisheye_filename)

        if not generate_fisheye_image(image_paths, fisheye_path, temp_dir):
            raise Exception("生成鱼眼图像失败")

        svf = calculate_svf(fisheye_path, svf_rings)

        result.update({
            "success": True,
            "fisheye_path": fisheye_path,
            "svf": svf
        })

        logging.info(f"坐标 {coord_key} 处理完成，SVF: {svf}")

    except Exception as e:
        result["error"] = str(e)
        logging.error(f"处理坐标 {coord_key} 失败: {str(e)}")

    return result

async def fetch_street_view_images(csv_path, output_dir, mask_dir):
    """
    从 CSV 文件中读取坐标，并爬取街景图片和天空掩膜。

    :param csv_path: CSV 文件路径
    :param output_dir: 图片保存目录
    :param mask_dir: 天空掩膜保存目录
    """
    data = read_csv(csv_path)
    if not data:
        logging.error("CSV 文件为空或无法读取。")
        return

    # 去掉 header
    data = data[1:]
    headings = ['0', '90', '180', '270']
    pitch = '30'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for row in data:
        try:
            # 从 CSV 读取的值是字符串，转为 float 后再进行坐标转换
            wgs_x = float(row[0])
            wgs_y = float(row[1])
            bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
            svid = getPanoId(bd09mc_x, bd09mc_y)

            if not svid:
                logging.warning(f"坐标 ({wgs_x}, {wgs_y}) 无法获取 svid，跳过。")
                continue

            for heading in headings:
                save_path = os.path.join(output_dir, f"{wgs_x}_{wgs_y}_{heading}_{pitch}.png")
                mask_path = os.path.join(mask_dir, f"mask_{wgs_x}_{wgs_y}_{heading}_{pitch}.png")
                url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={heading}&pitch={pitch}&width=480&height=320'
                img = grab_img_baidu(url)

                if img:
                    with open(save_path, "wb") as f:
                        f.write(img)

                    discard_image_regions(save_path, save_path, left_ratio=0.25, right_ratio=0.25)
                    logging.info(f"成功保存并裁剪图片: {save_path}")

                    # 提取天空掩膜
                    extract_and_save_sky_mask(save_path, mask_path)
                else:
                    logging.warning(f"图片抓取失败: {save_path}")

        except Exception as e:
            logging.error(f"处理坐标 ({row[0]}, {row[1]}) 时出错: {e}")

def discard_image_regions(image_path, output_path, left_ratio=0.25, right_ratio=0.25):
    """
    丢弃图片的左右部分内容，仅保留中间区域，输出图片宽度变小。

    :param image_path: 原始图片路径
    :param output_path: 裁剪后图片保存路径
    :param left_ratio: 左侧丢弃比例
    :param right_ratio: 右侧丢弃比例
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            # 计算保留区域
            left = int(width * left_ratio)
            right = width - int(width * right_ratio)
            # 只保留中间区域，直接裁剪掉两侧
            cropped_img = img.crop((left, 0, right, height))
            cropped_img.save(output_path)

            logging.info(f"图片裁剪成功: {output_path}")
    except Exception as e:
        logging.error(f"裁剪图片失败: {image_path}, 错误: {e}")

async def main():
    csv_path = "A100.csv"  # CSV 文件路径
    output_dir = "pictures"  # 图片保存目录
    mask_dir = "sky_masks"   # 天空掩膜目录


    # 爬取街景图片
    await fetch_street_view_images(csv_path, output_dir, mask_dir)

    # 查找所有掩膜图片，分组
    coordinate_groups = {}
    for file_name in os.listdir(mask_dir):
        if file_name.endswith(".png") and file_name.startswith("mask_"):
            parts = file_name.replace("mask_", "").replace(".png", "").split("_")
            if len(parts) == 4:
                wgs_x, wgs_y, heading, pitch = parts
                coord_key = f"{wgs_x}_{wgs_y}"
                if coord_key not in coordinate_groups:
                    coordinate_groups[coord_key] = {"wgs_x": wgs_x, "wgs_y": wgs_y, "images": {}}
                coordinate_groups[coord_key]["images"][heading] = os.path.join(mask_dir, file_name)

    # 批量处理，收集svf结果
    tasks = []
    coord_keys = []
    for coord_key, group in coordinate_groups.items():
        image_paths = group["images"]
        if len(image_paths) == 4:
            tasks.append(process_coordinate(coord_key, image_paths, "fisheye_output", tempfile.gettempdir(), svf_rings=39))
            coord_keys.append(coord_key)
        else:
            logging.warning(f"坐标 {coord_key} 掩膜图片不完整，跳过处理")

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 输出坐标和svf到csv
    csv_rows = []
    for idx, result in enumerate(results):
        if isinstance(result, dict) and result.get("success"):
            wgs_x = coordinate_groups[coord_keys[idx]]["wgs_x"]
            wgs_y = coordinate_groups[coord_keys[idx]]["wgs_y"]
            svf = result.get("svf", "")
            csv_rows.append([wgs_x, wgs_y, svf])
        else:
            logging.error(f"处理失败: {result}")

    # 写入csv
    with open('coordinate_svf_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['wgs_x', 'wgs_y', 'svf'])
        writer.writerows(csv_rows)

    logging.info(f"总计: {len(results)} 个坐标组，成功: {len(csv_rows)} 个，失败: {len(results) - len(csv_rows)} 个")

if __name__ == "__main__":
    asyncio.run(main())
