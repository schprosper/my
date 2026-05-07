改进说明

本项目在原有车牌识别流程基础上，针对普通蓝牌和新能源绿牌的差异进行了优化，主要改进包括：车牌定位优化、车牌类型判断、字符分割优化和模板匹配范围限制。

车牌定位优化：颜色优先，边缘兜底

原流程主要依靠 Sobel 边缘检测和轮廓筛选定位车牌。改进后，程序优先使用 HSV 颜色空间提取蓝色和绿色区域，用于快速定位普通蓝牌和新能源绿牌。

核心代码如下：

def locate_plate_by_color(image):
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

green_mask = cv2.inRange(hsv, np.array([35, 35, 40]), np.array([100, 255, 255]))
blue_mask = cv2.inRange(hsv, np.array([95, 50, 40]), np.array([135, 255, 255]))

mask = cv2.bitwise_or(green_mask, blue_mask)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
return choose_plate_rect(contours, image.shape)

在 getPlate() 中，程序会先尝试颜色定位：

color_rect = locate_plate_by_color(rawImage)
if color_rect is not None:
x, y, weight, height = color_rect
return crop_with_padding(rawImage, x, y, weight, height)

如果颜色定位失败，程序继续执行灰度化、Sobel 边缘检测、形态学处理和轮廓筛选，从而形成“颜色优先，边缘兜底”的两级定位策略，提高车牌定位的速度和稳定性。

车牌类型判断：区分 7 位蓝牌和 8 位绿牌

普通蓝牌通常按 7 位字符处理，新能源绿牌通常按 8 位字符处理。为了避免使用同一套分割规则导致漏切或多切，程序通过绿色像素占比判断是否为新能源车牌。

核心代码如下：

def is_new_energy_plate(image):
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv, np.array([35, 35, 40]), np.array([100, 255, 255]))
green_ratio = cv2.countNonZero(green_mask) / (image.shape[0] * image.shape[1])
return green_ratio > 0.25

def infer_plate_char_count(plate_image):
return 8 if plate_image is not None and is_new_energy_plate(plate_image) else 7

这样程序可以根据车牌类型自动决定期望字符数：绿牌按 8 位切割，蓝牌按 7 位切割，减少字符漏切、多切和错切。

字符分割优化：投影分割优先，轮廓分割兜底

字符分割阶段优先使用投影分割方法。如果投影分割能够得到符合预期数量的字符，就直接返回结果；如果失败，则使用轮廓检测方法作为备用方案。

核心代码如下：

def splitPlate(image, expected_count=None, plate_image=None):
if expected_count is None:
expected_count = infer_plate_char_count(plate_image)

projection_chars = splitPlateByProjection(plate_image, expected_count) if plate_image is not None else []
if projection_chars:
    return projection_chars

contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

对于新能源车牌，程序还会处理第 2 位后面的分隔点，避免将分隔点误切成字符。

核心代码如下：

def remove_plate_separator(zones, expected_count):
if len(zones) <= expected_count or len(zones) <= 2:
return zones

widths = np.array([x2 - x1 + 1 for x1, x2 in zones])
median_w = np.median(widths)

if widths[2] < median_w * 0.75:
    zones = zones[:2] + zones[3:]

return zones

4. 字符识别优化：根据位置限制模板匹配范围

原来的模板匹配容易让每一位字符都和全部模板比较，导致相似字符误识别。改进后，程序根据车牌字符位置限制模板范围：首位匹配省份简称，第二位匹配字母，后续位置按规则匹配字母或数字。

核心代码如下：

def getAllowedTemplateIndexes(position, total):
province_indexes = range(34, 68)
letter_indexes = range(10, 34)
number_indexes = range(0, 10)
letter_or_number_indexes = range(0, 34)

if position == 0:
    return province_indexes
if position == 1:
    return letter_indexes
if total == 8 and position >= 6:
    return number_indexes
if total == 7 and 2 <= position <= 3:
    return letter_indexes
if total == 7 and position >= 4:
    return number_indexes

return letter_or_number_indexes

在模板匹配时，程序只遍历当前位置允许的模板：

for index in getAllowedTemplateIndexes(char_index, len(plates)):
words = chars[index]

这样可以减少无效匹配范围，降低数字和字母之间的误识别。

总结

本次改进的核心思路是：将中国车牌的颜色特征、位数规则和字符位置规则加入识别流程中。定位阶段采用“颜色优先，边缘兜底”；分割阶段根据蓝牌和绿牌选择 7 位或 8 位规则；识别阶段根据字符位置限制模板范围。通过这些改进，系统能够同时处理普通蓝牌和新能源绿牌，提高了车牌定位、字符分割和模板匹配的准确性与稳定性。