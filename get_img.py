'''coding: utf-8
    get 某网站的数独图片
    现在可能没用了，在文件夹中有存
'''

#获取图片
import requests
import os
import random
number = random.randint(1,90)
print(number)
img = "image/"+ str(number) +'.jpg' ##目录根据自己自行更改
url = "http://psapi.gdieee.com/image/"+ str(number)

try:
    if not os.path.exists(img):
        #文件不存在
        image = requests.get(url)
        if image.status_code == 200:#请求正常
            with open(img,'wb') as f:
                f.write(image.content)
                print("Successfully save")
        else:
            print(image.status_code)
    else:
        print("the file has existed")
except Exception as e:
    print(e)