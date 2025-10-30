import os
import asyncio
import logging
import concurrent.futures
from io import BytesIO
import aiohttp
from fisheye import generate_fisheye_image
from svf import calculate_svf
from baiduStreetViewSpider import read_csv, getPanoId, wgs2bd09mc
from PIL import Image
from sky_segmentation import extract_sky_mask_from_image
import csv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def fetch_image(session, url, semaphore, headers=None):
    """使用 aiohttp 异步抓取图片，返回 bytes 或 None"""
    async with semaphore:
        try:
            async with session.get(url, headers=headers, timeout=20) as resp:
                if resp.status == 200:
                    content_type = resp.headers.get('Content-Type', '')
                    data = await resp.read()
                    if data and 'image' in content_type:
                        return data
                logging.warning(f"请求失败 {url} 状态码 {resp.status}")
        except Exception as e:
            logging.error(f"请求图片出错 {url}: {e}")
    return None


def discard_image_regions_pil(pil_img, left_ratio=0.25, right_ratio=0.25):
    """对 PIL.Image 在内存中裁剪左右两侧并返回新的 PIL.Image"""
    width, height = pil_img.size
    left = int(width * left_ratio)
    right = width - int(width * right_ratio)
    return pil_img.crop((left, 0, right, height))


async def process_coordinate(coord_key, wgs_x, wgs_y, svid, session, semaphore, process_pool, thread_pool, output_dir, svf_rings=39):
    """异步处理单个坐标：抓取四方向图、提取天空掩码、合成鱼眼并计算 SVF。只保存最终鱼眼图到磁盘。"""
    result = {"coordinate": coord_key, "success": False, "fisheye_path": None, "svf": None}
    headings = ['0', '90', '180', '270']
    pitch = '30'
    try:
        logging.info(f"[async] 开始处理 {coord_key}")
        # 并发抓取四张图片
        url_template = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={}&heading={}&pitch={}&width=480&height=320'
        tasks = [fetch_image(session, url_template.format(svid, h, pitch), semaphore) for h in headings]
        images_bytes = await asyncio.gather(*tasks)

        # 检查是否都抓取成功
        if not all(images_bytes):
            logging.warning(f"坐标 {coord_key} 有图片抓取失败，跳过")
            return result

        # 在内存中裁剪并提取天空掩码
        masks = {}
        for h, img_b in zip(headings, images_bytes):
            pil_img = Image.open(BytesIO(img_b)).convert('RGB')
            cropped = discard_image_regions_pil(pil_img, left_ratio=0.25, right_ratio=0.25)
            sky_mask = extract_sky_mask_from_image(cropped)
            if sky_mask is None:
                logging.warning(f"坐标 {coord_key} 方向 {h} 未检测到天空，跳过")
                return result
            masks[h] = sky_mask

        # 合成鱼眼（在线程池中执行以避免阻塞事件循环）
        os.makedirs(output_dir, exist_ok=True)
        fisheye_path = os.path.join(output_dir, f"fisheye_{coord_key}.png")

        loop = asyncio.get_running_loop()
        ok = await loop.run_in_executor(thread_pool, generate_fisheye_image, masks, fisheye_path, None)
        if not ok:
            logging.error(f"鱼眼合成失败: {coord_key}")
            return result

        # 提交 SVF 到进程池执行
        svf = await loop.run_in_executor(process_pool, calculate_svf, fisheye_path, svf_rings)

        result.update({"success": True, "fisheye_path": fisheye_path, "svf": svf})
        logging.info(f"[async] 完成 {coord_key} SVF={svf}")
        return result

    except Exception as e:
        logging.error(f"处理坐标 {coord_key} 异常: {e}")
        result['error'] = str(e)
        return result


async def main_async(csv_path='A100.csv', output_dir='fisheye_output', max_conn=20):
    data = read_csv(csv_path)
    if not data:
        logging.error('CSV 文件为空或无法读取。')
        return
    data = data[1:]

    cpu_count = max(1, (os.cpu_count() or 2) - 1)
    thread_workers = min(16, max(4, cpu_count * 2))

    logging.info(f"使用线程池 {thread_workers}（用于合成）和进程池 {cpu_count}（用于 SVF）")

    connector = aiohttp.TCPConnector(limit_per_host=8)
    semaphore = asyncio.Semaphore(max_conn)

    results = []
    async with aiohttp.ClientSession(connector=connector) as session:
        # 使用进程池和线程池
        with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as process_pool:
            with concurrent.futures.ThreadPoolExecutor(max_workers=thread_workers) as thread_pool:
                tasks = []
                for row in data:
                    try:
                        wgs_x = float(row[0])
                        wgs_y = float(row[1])
                    except Exception:
                        logging.warning(f"CSV 行数据无效，跳过: {row}")
                        continue

                    bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
                    # getPanoId 是阻塞的 requests 调用，交给线程池执行
                    loop = asyncio.get_running_loop()
                    svid = await loop.run_in_executor(None, getPanoId, bd09mc_x, bd09mc_y)
                    if not svid:
                        logging.warning(f"坐标 ({wgs_x}, {wgs_y}) 无法获取 svid，跳过。")
                        continue

                    coord_key = f"{wgs_x}_{wgs_y}"
                    task = asyncio.create_task(process_coordinate(coord_key, wgs_x, wgs_y, svid, session, semaphore, process_pool, thread_pool, output_dir))
                    tasks.append((coord_key, wgs_x, wgs_y, task))

                # collect results
                for coord_key, wgs_x, wgs_y, task in tasks:
                    res = await task
                    results.append((coord_key, wgs_x, wgs_y, res))

    # 写入结果 CSV
    csv_rows = []
    for coord_key, wgs_x, wgs_y, res in results:
        if res.get('success'):
            csv_rows.append([wgs_x, wgs_y, res.get('svf')])
        else:
            logging.warning(f"坐标 {coord_key} 处理失败: {res.get('error')})")

    with open('coordinate_svf_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['wgs_x', 'wgs_y', 'svf'])
        writer.writerows(csv_rows)

    logging.info(f"完成，共处理 {len(results)} 个坐标，成功 {len(csv_rows)} 个")


def main():
    asyncio.run(main_async())


if __name__ == '__main__':
    main()