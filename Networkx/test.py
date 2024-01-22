# https://blog.csdn.net/qq_35722520/article/details/106190232?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-106190232-blog-45391471.235%5Ev40%5Epc_relevant_3m_sort_dl_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-106190232-blog-45391471.235%5Ev40%5Epc_relevant_3m_sort_dl_base3&utm_relevant_index=5
import requests
import json
import pandas as pd


def download(cityname, line):
    url = 'https://restapi.amap.com/v3/bus/linename?s=rsv3&extensions=all&key=a8dc3d7e9d54044807dae7c235da218e&output=json&city={}&offset=1&keywords={}&platform=JS'.format(cityname, line)
    r = requests.get(url).text
    rt = json.loads(r)
    dt = {}
    dt['line_name'] = rt['buslines'][0]['name'].split('(')[0]
    station_name = []
    for st in rt['buslines'][0]['busstops']:
        station_name.append(st['name'])
    dt['station_name'] = station_name
    dm = pd.DataFrame(dt)
    dm.to_csv('{}{}公交基本信息.csv'.format(cityname, line), encoding='utf-8-sig')


if __name__ == '__main__':
    cityname = '北京'
    lines = ['311路']
    for line in lines:
        download(cityname, line)


# https://blog.csdn.net/gdhy9064/article/details/90070793
# import requests
# import time
#
# null = None #将json中的null定义为None
# city_code = 257 #广州的城市编号
# station_info = requests.get('http://map.baidu.com/?qt=bsi&c=%s&t=%s' % (
#                     city_code,
#                     int(time.time() * 1000)
#                )
# )
# station_info_json = eval(station_info.content)  #将json字符串转为python对象
# print(station_info_json)
