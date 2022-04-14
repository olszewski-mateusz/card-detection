import numpy as np
import skimage.morphology as mp
from skimage import feature, img_as_float, measure, transform
from skimage.color import rgb2gray
from skimage.segmentation import flood_fill
from skimage.transform import resize

class Error(Exception):
    pass

def prepareImage(image):
    gray = rgb2gray(image)
    MIN = np.percentile(gray, 0.0)
    MAX = np.percentile(gray, 100.0)
    norm = (gray - MIN) / (MAX - MIN)
    norm[norm[:,:] > 1] = 1
    norm[norm[:,:] < 0] = 0
    norm += 125 - np.mean(norm)
    
    can = feature.canny(norm, sigma=4.6)
    edges2 = mp.dilation(can, mp.disk(3))
    
    filled = flood_fill(img_as_float(edges2), (0,0), 0.5)
    filled[filled < 0.5] = 1.0
    filled[filled < 1.0] = 0.0
    return filled

def getCardsContours(img):
    contours = measure.find_contours(img, 0.8)
    selected = []
    for contour in contours:
        if len(contour[:,0]) >= 500:
            selected.append(contour)
    return selected

def findCorners(contour, step=20, alpha=0.04, beta=5):
    
    def get_intercept(point1, point2, tan1, tan2):
        delta_tan = tan1 - tan2
        if(delta_tan == 0):
            print("Błąd!")
        x1 = point1[0]
        x2 = point2[0]
        y1 = point1[1]
        y2 = point2[1]
        x = (tan1*x1 - tan2*x2 - y1 + y2) / delta_tan 
        y = (tan1*tan2*(x1-x2) + tan1*y2 - tan2*y1) / delta_tan 
        return y,x

    angle_dict = dict()
    for i in range(len(contour)-step):
        delta_x = contour[i,0] - contour[i+step,0]
        delta_y = contour[i,1] - contour[i+step,1]
        angle = np.arctan2(delta_x, delta_y) * 180 / np.pi
        index = round(angle*alpha)
        angle_dict.setdefault(index, []).append(angle);
    li = list(angle_dict.keys())
    li.sort(key = lambda x: len(angle_dict[x]), reverse=True)
    result = []
    for lii in li:
        for res in result:
            if abs(res - lii) == 1:
                break
        else:
            result.append(lii)
    if len(result) < 4:
        raise Error()
    
    li = result
    li = li[:4]
    li.sort() 
    keys = [np.mean(angle_dict[lii]) for lii in li]

    angle_dict = dict()
    x_dict = dict()
    y_dict = dict()
    for i in range(len(contour)-step):
        delta_x = contour[i,0] - contour[i+step,0]
        delta_y = contour[i,1] - contour[i+step,1]
        angle = np.arctan2(delta_x, delta_y) * 180 / np.pi
        for key_angle in keys:
            if abs(key_angle - angle) < beta:
                angle_dict.setdefault(key_angle, []).append(angle);
                x_dict.setdefault(key_angle, []).append(np.mean([contour[i,0],contour[i+step,0]]));
                y_dict.setdefault(key_angle, []).append(np.mean([contour[i,1],contour[i+step,1]]));    

    if len(list(angle_dict.keys())) < 4:
        raise Error()
    
    points = []
    for i in range(4):
        j = (i+1) % 4
        angle1 = -np.mean(angle_dict[keys[i]])
        x1 = np.mean(x_dict[keys[i]])
        y1 = np.mean(y_dict[keys[i]])
        angle2 = -np.mean(angle_dict[keys[j]])
        x2 = np.mean(x_dict[keys[j]])
        y2 = np.mean(y_dict[keys[j]])
        tan1 = np.tan(angle1 * np.pi / 180 -np.pi/2 )
        tan2 = np.tan(angle2 * np.pi / 180 -np.pi/2 )
        points.append(get_intercept([x1,y1], [x2,y2], tan1, tan2))
            
    points2 = points[1:] + [points[0]]
    
    path1 = (points[0][0]-points[1][0])**2 + (points[0][1]-points[1][1])**2
    path2 = (points[1][0]-points[2][0])**2 + (points[1][1]-points[2][1])**2

    if path1 > path2: 
        points, points2 = points2, points
    
    return points, points2

def warpCard(img, points): 
    src = np.array([[340, 500],[0, 500],[0, 0],[340, 0]])
    dst = np.array(points)
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    warped = transform.warp(img, tform, output_shape=(500, 340))
    return warped

def cutCard(img):
    return img[10:150,10:70]

def thresholdCard(img, q=0.2):
    gray = rgb2gray(img)
    binary = gray > np.quantile(gray, q)
    return binary

def prepareCard(img):
    max_x = len(img[0,:])
    max_y = len(img[:,0])
    for i in range(max_y):
        img = flood_fill(img_as_float(img), (i, 0), 0.5)
        img = flood_fill(img_as_float(img), (i, max_x-1), 0.5)
    for i in range(max_x):
        img = flood_fill(img_as_float(img), (0, i), 0.5)
        img = flood_fill(img_as_float(img), (max_y-1, i), 0.5)
    img[img < 0.5] = 1.0
    img[img < 1.0] = 0.0
    return img

def getSymbolsContours(img):
    contours = measure.find_contours(img, 0.8)
    selected = []
    for contour in contours:
        if len(contour[:,0]) >= 20:
            selected.append(contour)
    return selected

def cutSymbol(contour, image):
    new_image = image.copy()
    min_x = int(np.amin(contour[:, 1]))+1
    max_x = int(np.amax(contour[:, 1]))+1
    min_y = int(np.amin(contour[:, 0]))+1
    max_y = int(np.amax(contour[:, 0]))+1
    new_image = new_image[min_y:max_y, min_x:max_x]
    return new_image

def compareSymbolWithTemplate(img, template):
    img_resized = resize(img, (template.shape[0], template.shape[1]))
    img_resized[img_resized < 0.5] = 0
    img_resized[img_resized >= 0.5] = 255
    result = img_resized==template
    return result

def is_correct_match(upper_score, lower_score):
    if upper_score < 0.1 or lower_score < 0.1:
        return False
    else:
        return True
