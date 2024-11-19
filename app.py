import cv2
import numpy as np
import os

# 去重模板匹配法
def stitch2(img1, img2, ratio):
    # 如果没有传入图片
    if img1 is None or img2 is None:
        raise FileNotFoundError

    # 如果宽度不一样 直接不处理
    if img1.shape[1] != img2.shape[1]:
        raise ValueError

    # 考虑到旁边滑动条影响 左右个去处20像素进行匹配
    img1 = img1[0:img1.shape[0], 20:img1.shape[1] - 20, 0:img1.shape[2]]
    img2 = img2[0:img2.shape[0], 20:img2.shape[1] - 20, 0:img2.shape[2]]
    cv2.imwrite("result/去掉左右20像素的图一.jpg", img1)
    cv2.imwrite("result/去掉左右20像素的图二.jpg", img2)

    # 保存图片的尺寸信息
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    hdiff = h1-h2

    # 把图一高度裁剪到和图二一样
    img1 = img1[img1.shape[0] - img2.shape[0]:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]]

    # 如果是色彩图像，则将图像灰度化(3就是色彩图像)
    if img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
        gray2 = img2

    cv2.imwrite("result/灰度处理后的图一.jpg", gray1)
    cv2.imwrite("result/灰度处理后的图二.jpg", gray2)
    

    # 使用了 OpenCV 中的 自适应阈值处理，将灰度图像 gray 转换为 二值图像（即像素值为 0 或 255 的图像）
    # 这里为 255，所以像素值只有两种可能：0 和 255
    # 在这里选择的是 均值方法 (cv2.ADAPTIVE_THRESH_MEAN_C)
    # cv2.THRESH_BINARY：将大于阈值的像素设置为 maxValue，小于或等于阈值的像素设置为 0。
    # blockSize：定义邻域窗口的大小（必须是奇数），用于计算阈值。 这里是 11，即以 11x11 的窗口为中心计算局部阈值。
    # C：常数，用于微调阈值。这里是 2，表示从计算出的局部均值中减去 2，以获得更精确的阈值。
    # 返回一个高宽数组，每个像素的二值(要么是0要么死255)
    thresh1 = cv2.adaptiveThreshold(gray1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.adaptiveThreshold(gray2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("result/自适应阈值后的图1.jpg", thresh1)
    cv2.imwrite("result/自适应阈值后的图2.jpg", thresh2)

    # 第一步 找到两张图片底部相同的部分 (比如微信的聊天聊天输入框)
    # 计算两幅图像的二值化差异 (相同的地方减出来肯定是0)
    sub = thresh1 - thresh2
    # 消除噪声，平滑图像
    sub = cv2.medianBlur(sub, 3)
    cv2.imwrite("result/差异化计算后的图片.jpg", sub)

    # 把0和255改成0和1
    sub = sub // 255

    # 定义一个差异化的阈值 至少差异化要大于十分之一
    # 比如图片宽度为1000那么不同的像素至少要大于100个
    thresh = w1 // 10

    # 考虑到部分机型底部刘海问题 或者有滑动条等问题 去除掉底部3%的位置开始匹配
    height = h2
    min_height = h2*0.97
    print(height)
    # 从下到上对比(h越大也就是越底)
    for i in range(h2 - 1, 0, -1):
        # 判断当前行是否满足差异化 也就是10%的差异化
        # 并且高度必须小于97%(也就是去掉顶部的)
        if np.sum(sub[i]) > thresh and height < min_height:
            print(np.sum(sub[i]))
            break
        height = height - 1

    # 每一块的高度
    block = sub.shape[0] // ratio

    # 如果差异的高度比每一块的高度还要小 那就怎么两张图片几乎一样，不做处理
    if height < block:
        return 0, 1, 0


    # 图一的底部
    templ = gray1[height - block:height, ]
    cv2.imwrite("result/图一的底部.jpg", templ)

    # 如果高度小于一块的高度
    if templ.shape[0] < block:
        print("templ shape is too small", templ.shape)
        return 0, 1, 0

    # 在图二中去找图一的底部
    res = cv2.matchTemplate(gray2, templ, cv2.TM_SQDIFF_NORMED)
    # 返回的好像是每一行的相似度


    # mn_val和min_loc表示最佳匹配值和匹配位置
    mn_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("thresh, shape ,height, block", thresh, sub.shape, height, block)
    print("mn_val, max_val, min_loc, max_loc", mn_val, max_val, min_loc, max_loc)

    # 考虑到JPG 可能存在成像压缩，至少需要95的相似度才认为匹配成功
    if mn_val < 0.05:
        # result, bottom, top
        # result, 图一底部需要裁掉的部分, 图二顶部需要裁掉的部分
        return 1, (height - block + hdiff) / h1, min_loc[1] / h2
    else:
        return 0, 1, 0


def draw(img1, img2):
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape

    # result, 图一底部需要裁掉的部分, 图二顶部需要裁掉的部分
    result, bottom, top = stitch2(img1, img2, 15)
    if result == 0:
        return img1

    # 把比例转换为实际的高度
    bottom = int(bottom * h1)
    top = int(top * h2)

    roi1 = img1[0:bottom, ]
    roi2 = img2[top:h2, ]
    cv2.imwrite("result/图一裁掉剩余.jpg", roi1)
    cv2.imwrite("result/图二裁掉剩余.jpg", roi2)

    # 创建一个图片 (高度为图一剩余+图二剩余)，显式指定 dtype=np.uint8
    image = np.ones((roi1.shape[0] + roi2.shape[0], w1, 3), dtype=np.uint8)
    # 图片的第一部分
    image[0:roi1.shape[0], 0:w1, 0:3] = roi1
    # 图片的第二部分
    image[roi1.shape[0]:roi1.shape[0] + roi2.shape[0], 0:w1, 0:3] = roi2
    # 返回图片
    return image

import cv2
import os

def video2img(video_path, frame_interval=1):
    # 检查视频路径是否存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {video_path}")
    
    frame_count = 0  # 读取的帧数
    dst = None # 保存拼接结果

    while True:
        ret, frame = cap.read()
        if not ret:  # 视频读取完成
            break
        
        # 只保存符合间隔条件的帧
        if frame_count % frame_interval == 0:
            if dst is None:
                dst = frame
            else:
                dst = draw(dst, frame)
        
        frame_count += 1

    # 释放视频资源
    cap.release()
    return dst



if __name__ == "__main__":
    # 输出目录
    if not os.path.exists("result"):
        os.makedirs("result")

    # 图片拼接长图
    img1 = cv2.imread("./imgs/IMG_1.PNG")
    img2 = cv2.imread("./imgs/IMG_2.PNG")
    img3 = cv2.imread("./imgs/IMG_3.PNG")
    img4 = cv2.imread("./imgs/IMG_4.PNG")
    img5 = cv2.imread("./imgs/IMG_5.PNG")
    dst = None;
    for img in [img1, img2, img3, img4, img5]:
        if dst is None:
            dst = img
        else:
            dst = draw(dst, img)   
    cv2.imwrite("result/result.jpg", dst)

    # 视频拼接长图
    video_file = "./video/RPReplay_Final1732015583.MP4"
    frame_interval = 10  # 每10帧保存一次
    dst = video2img(video_file, frame_interval)
    cv2.imwrite("result/result1.jpg", dst)

