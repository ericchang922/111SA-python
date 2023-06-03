import cv2
import numpy as np
import time
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


ambulance_classifier = cv2.CascadeClassifier('ambulance.xml')
cap = cv2.VideoCapture('./img/videoooooooo.mp4')
@app.route('/road-view', methods=['GET'])
def roadVeiw():
    x_left = 500
    x_right = 1500
    min_w = 180
    min_h = 180
    line_high = 800
    emergency = determine(cap, x_right, x_left, min_w, min_h, line_high)
    return jsonify(
        {
            "statusCode": 200,
            "response": {"emergency":emergency}
        }
    )


def determine(cap, x_right, x_left, min_w, min_h, line_high):
    decide = True
    while decide:
        bgsubmog = cv2.bgsegm.createBackgroundSubtractorMOG()

        start = time.time()

        if cap is None:
            print("路径问题")
        else:
            while True:
                # 读取视频帧
                ret, frame = cap.read()

                # cv2.putText(frame, "Cars Count:" + str(carno), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                         #   (0, 255, 0), 2)

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
                    cv2.line(frame, (x_left, line_high), (x_right, line_high), (255, 255, 8), 3)
                    # 取出轮廓点绘图
                    for (i, c) in enumerate(cnts):
                        x, y, w, h = cv2.boundingRect(c)
                        # 对车辆的宽高进行判断，验证是否是有效的车辆
                        isValid = (w >= min_w) & (h >= min_h)
                        if (not isValid):
                            continue
                        # 得到有效车辆信息
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        # # 计算有效车辆的中心点
                        # cpoint = center(x, y, w, h)
                        # cars.append(cpoint)
                        # (x, y) = cpoint

                        # 救护车检测
                        roi = frame[y:y + h, x:x + w]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        ambulances = ambulance_classifier.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

                        # 辨識救護車，但框出來的不是真正的
                        # 如果辨識到救護車會直接回傳carno=9999，這樣就能知道是否有緊急狀況
                        for (ax, ay, aw, ah) in ambulances:
                            # 計算救護車中心點位置
                            ax += x
                            ay += y
                            if (aw > min_w and ah > min_h):
                                cv2.rectangle(frame, (ax, ay), (ax + aw, ay + ah), (0, 0, 255), 2)
                                #decide = False
                                return decide

                    cv2.namedWindow("frame", 0)  # 0可调大小，注意：窗口名必须imshow里面的一窗口名一直
                    cv2.resizeWindow("frame", 1200, 800)  # 设置长和宽
                    cv2.imshow('frame', frame)

                key = cv2.waitKey(1)

                # if (key == 27 or ((time.time()) - start) >= 20.0):
                #     print(("end:" + str(time.time())))
                #     break

        # 释放缓存资源
        #cap.release()
        # 释放所有窗口
        #cv2.destroyAllWindows()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)