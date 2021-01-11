# 导入工具包
from imutils import contours
import numpy as np
import argparse
import cv2
# 对轮廓进行排序


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0

    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 计算外接矩形 boundingBoxes是一个元组
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]  # 用一个最小的矩形，把找到的形状包起来x,y,h,w
    # sorted排序
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes  # 轮廓和boundingBoxess


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


# 主函数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='./images/credit_card_01.png',
                help="path to input image")
ap.add_argument("-t", "--template", default='./ocr_a_reference.png',
                help="path to template OCR image")
args = vars(ap.parse_args())

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 读取一个模板图像
img = cv2.imread(args["template"])
# cv_show('template',img)
# 灰度图
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv_show('template_gray',ref)
# 二值图像
_,ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)
cv_show('template_bi', ref)

# 【二、模板处理流程】: 轮廓检测, 外接矩形, 抠出模板, 让模板对应每个数值
# 1.计算轮廓
'''
cv2.findContours()函数接受的参数为二值图，即黑白的（不是灰度图）,
	cv2.RETR_EXTERNAL只检测外轮廓，
	cv2.CHAIN_APPROX_SIMPLE只保留终点坐标
	返回的list中每个元素都是图像中的一个轮廓
'''
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
img = cv2.drawContours(img, refCnts, -1, (0, 255, 0), 2)  # 轮廓在二值图上得到, 画是画在原图上
cv_show('template_Contours', img)
print(np.array(refCnts).shape)  # 10个轮廓,所以是10
refCnts = sort_contours(refCnts, method="left-to-right")[0]
digits = {}
# 2.遍历每一个轮廓,外接矩形
for (i, c) in enumerate(refCnts):  # c是每个轮廓的终点坐标
    # 计算外接矩形并且resize成合适大小
    (x, y, w, h) = cv2.boundingRect(c)
    # 3.抠出模板
    roi = ref[y:y + h, x:x + w]  # 每个roi对应一个数字
    # print(roi.shape)
    roi = cv2.resize(roi, (57, 88))  # 太小,调大点

    # 4.每一个数字对应每一个模板
    digits[i] = roi
    # cv2.imshow('roi_'+str(i),roi)
    # cv2.waitKey(0)
    # print(digits)

# 【三、输入图像处理】
# 形态学操作,礼帽+闭操作可以突出明亮区域,但并不是非得礼帽+闭操作

# 1.初始化卷积核,根据实际任务指定大小,不一定非要3x3
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
onekernel = np.ones((9, 9), np.uint8)

# 2.读取输入图像，预处理
image = cv2.imread(args["image"])
# cv_show('Input_img',image)
image = resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv_show('Input_gray',gray)

# 3.礼帽操作，突出更明亮的区域
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
# cv_show('Input_tophat',tophat)
# 4.x方向的Sobel算子,实验表明,加y的效果的并不好
gradX = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)  # ksize=-1相当于用3*3的

gradX = np.absolute(gradX)  # absolute: 计算绝对值
min_Val, max_val = np.min(gradX), np.max(gradX)
gradX = (255 * (gradX - min_Val) / (max_val - min_Val))
gradX = gradX.astype("uint8")

print(np.array(gradX).shape)
# cv_show('Input_Sobel_gradX',gradX)

# 5.通过闭操作（先膨胀，再腐蚀）将数字连在一起.  将本是4个数字的4个框膨胀成1个框,就腐蚀不掉了
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# cv_show('Input_CLOSE_gradX',gradX)

# 6.THRESH_OTSU会自动寻找合适的阈值，适合双峰，需把阈值参数设置为0
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv_show('Input_thresh',thresh)

# 7.再来一个闭操作,填补空洞
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
# cv_show('Input_thresh_CLOSE',thresh)

# 8.计算轮廓
threshCnts = cv2.findContours(thresh.copy(),
                              cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)[0]
cur_img = image.copy()
cv2.drawContours(cur_img, threshCnts, -1, (0, 0, 255), 2)
# cv_show('Input_Contours',cur_img)

# 【四、遍历轮廓和数字】
# 1.遍历轮廓
locs = []  # 存符合条件的轮廓
for i, c in enumerate(threshCnts):
    # 计算矩形
    x, y, w, h = cv2.boundingRect(c)

    ar = w / float(h)
    # 选择合适的区域，根据实际任务来，这里的基本都是四个数字一组
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 符合的留下来
            locs.append((x, y, w, h))

# 将符合的轮廓从左到右排序
locs = sorted(locs, key=lambda x: x[0])

# 2.遍历每一个轮廓中的数字
output = []  # 存正确的数字
for (i, (gx, gy, gw, gh)) in enumerate(locs):  # 遍历每一组大轮廓(包含4个数字)
    # initialize the list of group digits
    groupOutput = []

    # 根据坐标提取每一个组(4个值)
    group = gray[gy - 5:gy + gh + 5, gx - 5:gx + gw + 5]  # 往外扩一点
    cv_show('group_' + str(i), group)
    # 2.1 预处理
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]  # 二值化的group
    # cv_show('group_'+str(i),group)
    # 计算每一组的轮廓 这样就分成4个小轮廓了
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 排序
    digitCnts = sort_contours(digitCnts, method="left-to-right")[0]

    # 2.2 计算并匹配每一组中的每一个数值
    for c in digitCnts:  # c表示每个小轮廓的终点坐标
        z = 0
        # 找到当前数值的轮廓,resize成合适的的大小
        (x, y, w, h) = cv2.boundingRect(c)  # 外接矩形
        roi = group[y:y + h, x:x + w]		# 在原图中取出小轮廓覆盖区域,即数字
        roi = cv2.resize(roi, (57, 88))
        # cv_show("roi_"+str(z),roi)

        # 计算匹配得分: 0得分多少,1得分多少...
        scores = []  # 单次循环中,scores存的是一个数值 匹配 10个模板数值的最大得分

        # 在模板中计算每一个得分
        # digits的digit正好是数值0,1,...,9;digitROI是每个数值的特征表示
        for (digit, digitROI) in digits.items():
            # 进行模板匹配, res是结果矩阵
            # 此时roi是X digitROI是0 依次是1,2.. 匹配10次,看模板最高得分多少
            res = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            Max_score = cv2.minMaxLoc(res)[1]  # 返回4个,取第二个最大值Maxscore
            scores.append(Max_score)  # 10个最大值
        # print("scores：",scores)
        # 得到最合适的数字
        groupOutput.append(str(np.argmax(scores)))  # 返回的是输入列表中最大值的位置
        z = z + 1
    # 2.3 画出来
    cv2.rectangle(image, (gx - 5, gy - 5), (gx + gw + 5, gy + gh + 5), (0, 0, 255), 1)  # 左上角,右下角
    # 2.4 putText参数：图片,添加的文字,左上角坐标,字体,字体大小,颜色,字体粗细
    cv2.putText(image, "".join(groupOutput), (gx, gy - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    # 2.5 得到结果
    output.extend(groupOutput)
    print("groupOutput:", groupOutput)
    # cv2.imshow("Output_image_"+str(i), image)
    # cv2.waitKey(0)
# 3.打印结果
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Output_image", image)
cv2.waitKey(0)
