import re, os
import json
import requests
import time, glob
import csv
import traceback


# read csv
def write_csv(filepath, data, head=None):
    if head:
        data = [head] + data
    with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)


# write csv
def read_csv(filepath):
    data = []
    if os.path.exists(filepath):
        with open(filepath, mode='r', encoding='utf-8') as f:
            lines = csv.reader(f)  # #此处读取到的数据是将每行数据当做列表返回的
            for line in lines:
                data.append(line)
        return data
    else:
        print('filepath is wrong：{}'.format(filepath))
        return []


def grab_img_baidu(_url, _headers=None):
    if _headers == None:
        # 设置请求头 request header
        headers = {
            "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="90", "Google Chrome";v="90"',
            "Referer": "https://map.baidu.com/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
        }
    else:
        headers = _headers
    response = requests.get(_url, headers=headers)

    if response.status_code == 200 and response.headers.get('Content-Type') == 'image/jpeg':
        return response.content
    else:
        return None


def openUrl(_url):
    # 设置请求头 request header
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36 Edg/139.0.0.0"
    }
    response = requests.get(_url, headers=headers)
    if response.status_code == 200:  # 如果状态码为200，寿命服务器已成功处理了请求，则继续处理数据
        return response.content
    else:
        return None


def getPanoId(_lng, _lat):
    # 获取百度街景中的svid get svid of baidu streetview
    url = "https://mapsv0.bdimg.com/?&qt=qsdata&x=%s&y=%s&l=17.031000000000002&action=0&mode=day&t=1530956939770" % (
        str(_lng), str(_lat))
    response = openUrl(url).decode("utf8")
    # print(response)
    if (response == None):
        return None
    reg = r'"id":"(.+?)",'
    pat = re.compile(reg)
    try:
        svid = re.findall(pat, response)[0]
        return svid
    except:
        return None



def wgs84_to_gcj02(lng, lat):
    import math
    def out_of_china(lng, lat):
        return not (73.66 < lng < 135.05 and 3.86 < lat < 53.55)
    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + 0.1 * lng * lat + 0.2 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * math.pi) + 40.0 * math.sin(lat / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * math.pi) + 320 * math.sin(lat * math.pi / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + 0.1 * lng * lat + 0.1 * math.sqrt(abs(lng))
        ret += (20.0 * math.sin(6.0 * lng * math.pi) + 20.0 * math.sin(2.0 * lng * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * math.pi) + 40.0 * math.sin(lng / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * math.pi) + 300.0 * math.sin(lng / 30.0 * math.pi)) * 2.0 / 3.0
        return ret
    if out_of_china(lng, lat):
        return lng, lat
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return mglng, mglat

def gcj02_to_bd09(lng, lat):
    import math
    z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * math.pi)
    theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * math.pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat

def bd09_to_bd09mc(lng, lat):
    # 百度官方LL2MC参数
    LL2MC = [
        [ -0.0015702102444, 111320.7020616939, 1704480524535203, -10338987376042340, 26112667856603880, -35149669176653700, 26595700718403920, -10725012454188240, 1800819912950474, 82.5 ],
        [ 0.0008277824516172526, 111320.7020463578, 647795574.6671607, -4082003173.641316, 10774905663.51142, -15171875531.51559, 12053065338.62167, -5124939663.577472, 913311935.9512032, 67.5 ],
        [ 0.00337398766765, 111320.7020202162, 4481351.045890365, -23393751.19931662, 79682215.47186455, -115964993.2797253, 97236711.15602145, -43661946.33752821, 8477230.501135234, 52.5 ],
        [ 0.00220636496208, 111320.7020209128, 51751.86112841131, 3796837.749470245, 992013.7397791013, -1221952.21711287, 1340652.697009075, -620943.6990984312, 144416.9293806241, 37.5 ],
        [ -0.0003441963504368392, 111320.7020576856, 278.2353980772752, 2485758.690035394, 6070.750963243378, 54821.18345352118, 9540.606633304236, -2710.55326746645, 1405.483844121726, 22.5 ],
        [ -0.0003218135878613132, 111320.7020701615, 0.00369383431289, 823725.6402795718, 0.46104986909093, 2351.343141331292, 1.58060784298199, 8.77738589078284, 0.37238884252424, 7.45 ]
    ]
    LLBAND = [75, 60, 45, 30, 15, 0]
    abs_lat = abs(lat)
    for i in range(len(LL2MC)):
        if abs_lat >= LLBAND[i]:
            c = LL2MC[i]
            break
    else:
        c = LL2MC[-1]
    x = c[0] + c[1] * abs(lng)
    y = abs(lat) / c[9]
    y = c[2] + c[3] * y + c[4] * y ** 2 + c[5] * y ** 3 + c[6] * y ** 4 + c[7] * y ** 5 + c[8] * y ** 6
    x *= (-1 if lng < 0 else 1)
    y *= (-1 if lat < 0 else 1)
    return x, y

def wgs2bd09mc(lng, lat):
    lng, lat = wgs84_to_gcj02(lng, lat)
    lng, lat = gcj02_to_bd09(lng, lat)
    lng, lat = bd09_to_bd09mc(lng, lat)
    return lng, lat


if __name__ == "__main__":
    root = r'.\dir'
    read_fn = r'point_coordinate_intersect.csv'
    error_fn = r'error_road_intersection.csv'
    dir = r'images'
    filenames_exist = glob.glob1(os.path.join(root, dir), "*.png")

    # 读取 csv 文件
    data = read_csv(os.path.join(root, read_fn))
    # 记录 header
    header = data[0]
    # 去掉 header
    data = data[1:]
    # 记录爬取失败的图片
    error_img = []
    # 记录没有svid的位置
    svid_none = []
    headings = ['0', '90', '180', '270'] # directions, 0 is north
    pitchs = '0'

    count = 1
    # while count < 210:
    for i in range(len(data)):
        print('Processing No. {} point...'.format(i + 1))
        # gcj_x, gcj_y, wgs_x, wgs_y = data[i][0], data[i][1], data[i][2], data[i][3]
        wgs_x, wgs_y = float(data[i][15]), float(data[i][16])

        try:
            bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
        except Exception as e:
            print(str(e))  # 抛出异常的原因
            continue
        flag = True
        for k in range(len(headings)):
            flag = flag and "%s_%s_%s_%s.png" % (wgs_x, wgs_y, headings[k], pitchs) in filenames_exist

        # If all four files exist, skip
        if (flag):
            continue
        svid = getPanoId(bd09mc_x, bd09mc_y)
        print(svid)
        for h in range(len(headings)):
            save_fn = os.path.join(root, dir, '%s_%s_%s_%s.png' % (wgs_x, wgs_y, headings[h], pitchs))
            url = 'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={}&heading={}&pitch=0&width=480&height=320'.format(
                svid, headings[h]
            )
            img = grab_img_baidu(url)
            if img == None:
                data[i].append(headings[h])
                error_img.append(data[i])

            if img != None:
                # print(os.path.join(root, dir))
                with open(os.path.join(root, dir) + r'\%s_%s_%s_%s.png' % (wgs_x, wgs_y, headings[h], pitchs),
                          "wb") as f:
                    f.write(img)

        # 记得睡眠6s，太快可能会被封
        time.sleep(6)
        count += 1
    # 保存失败的图片
    if len(error_img) > 0:
        write_csv(os.path.join(root, error_fn), error_img, header)
