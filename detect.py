import cv2
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent import pywsgi

# 判断是否是车辆的最小矩形
min_w = 180
min_h = 180
# 检测线的高度
line_high = 800
# 统计有效车的数组
cars = []
# 线的偏移量
offset = 8
# 统计数量
carno = 0

ambul = 0

ambulance_classifier = cv2.CascadeClassifier('haarcascade_smile.xml')

app = Flask(__name__)
CORS(app)

#@app.route('/')
#def convey():
#    return jsonify(carno)

@app.route('/road-veiw', methods=['POST'])
def roadVeiw():
    insertValue = request.get_json()
    roadId = insertValue['roadId']
    #data = {
    #    "carNo": carno,
    #    "roadLane": 0,
    #    "roadLength": 0
    #}
    return jsonify({"carno":carno})


# 计算中心点函数
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


cap = cv2.VideoCapture('./img/videoooooooo.mp4')
# 引入去背景函数
bgsubmog = cv2.bgsegm.createBackgroundSubtractorMOG()

start = time.time()

if cap is None:
    print("路径问题")
else:
    while True:

        # 读取视频帧
        ret, frame = cap.read()

        cv2.putText(frame, "Cars Count:" + str(carno), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 0), 2)

        # print(frame.shape)
        if (ret == True):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 去除背景
            blur = cv2.GaussianBlur(gray, (7, 7), sigmaX=5)
            mask = bgsubmog.apply(blur)
            # 卷积核
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            # 腐蚀操作-去除背景中较小的噪点
            erode = cv2.erode(mask, kernel, iterations=2)
            # 膨胀操作：还原放大车辆
            dilate = cv2.dilate(erode, kernel2, iterations=1)
            # 闭运算-填补车辆像素空隙
            close1 = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel2)
            # close2 = cv2.morphologyEx(close1,cv2.MORPH_CLOSE,kernel2)
            # 发现轮廓
            cnts, hi = cv2.findContours(close1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 检测线
            cv2.line(frame, (500, line_high), (1500, line_high), (255, 255, 8), 3)
            # 取出轮廓点绘图
            for (i, c) in enumerate(cnts):
                x, y, w, h = cv2.boundingRect(c)
                # 对车辆的宽高进行判断，验证是否是有效的车辆
                isValid = (w >= min_w) & (h >= min_h)
                if (not isValid):
                    continue
                #print("y:" + str(y))
                #print("x:" + str(x))
                # 得到有效车辆信息
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # 计算有效车辆的中心点
                cpoint = center(x, y, w, h)
                cars.append(cpoint)
                (x, y) = cpoint
                #print("cpoint:" + str(cpoint))

                # 救护车检测
                roi = frame[y:y + h, x:x + w]
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #ambulances = 0
                ambulances = ambulance_classifier.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)
                #if(ambulances != 0):
                #    carno = 9999
                #else:

                #print(ambulances)

                for (ax, ay, aw, ah) in ambulances:
                    # 计算救护车中心点在整个图像中的坐标
                    ax += x
                    ay += y
                    cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 2)
                    # 统计救护车数量
                    ambul += 1

                for (x, y) in cars:
                    cal = y-line_high
                    if(cal < 0):
                        cal = cal * -1
                    if(cal < offset and x>400 and x<1500):
                        print("y:" + str(y))
                        print("x:" + str(x))
                        carno += 1
                        cars.remove((x, y))
                        print(carno)


            cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
            cv2.resizeWindow("frame", 1200, 800)  # 设置长和宽
            cv2.imshow('frame', frame)

            # cv2.imshow('dilate',dilate)


        #print("start:" + str(start))
        key = cv2.waitKey(1)


        if (key == 27 or ((time.time()) - start) >= 15.0 or ambul > 0):
            print(("end:" + str(time.time())))
            break

# 释放缓存资源
cap.release()
# 释放所有窗口
cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)

