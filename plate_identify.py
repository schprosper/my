import joblib
import numpy as np
import cv2
import copy

def plate_identify_SVM(contours,image):
    # 加载保存的SVM模型
    classifier = joblib.load('..\svm_蓝\IfPlate_svm_bule.joblib')

    for item in contours:
        rect = cv2.boundingRect(item)
        x = rect[0]
        y = rect[1]
        weight = rect[2]
        height = rect[3]
        img = image[y:y + height, x:x + weight]
        img = cv2.resize(img,(147,40))
        plate = img.copy()
        img = img.reshape(1, -1)
        img = np.array(img).astype(np.float32)
        predict = classifier.predict(img)
        if predict == 1:
            return plate
        else:
            continue


