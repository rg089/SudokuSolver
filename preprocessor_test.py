import cv2
import numpy as np
import imutils, os

debug = False
showDigitsOnly = False


def extract_digit(cell):
    """Cell is from the cleaned image."""
    if debug and not showDigitsOnly:
        imutils.showImage(cell)
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]
    hei, wid = thresh.shape[:2]
    minArea = 0.05*wid*hei
    maxArea = 0.9*wid*hei
    roi = thresh[int(0.3*hei): int(0.7*hei), int(0.3*wid):int(0.7*hei)]
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(cnts) == 0 or (roi>0).sum()/(roi.shape[0]*roi.shape[1])<0.02:
        if debug and not showDigitsOnly:
            thresh1 = thresh.copy()
            cv2.rectangle(thresh1, (int(0.3*wid), int(0.3*hei)), (int(0.7*wid), int(0.7*hei)), 255, 1)
            imutils.showImage(thresh1)
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        extr = thresh[y:y+h, x:x+w]
        if x == 0 or x+w == wid or w < 3 or h < 3 or w*h < minArea or (extr > 0).sum()/(wid*hei) < 0.0225 or w*h > maxArea:  # or not (x<=wid/2<=x+w and y<=hei/2<=y+h):
            if debug and not showDigitsOnly:
                print("Not detected. Details:", x, y, w, h, (extr > 0).sum()/(wid*hei)*100)
            continue
        if debug and not showDigitsOnly:
            cell1 = cell.copy()
            print(x,y,w,h)
            cv2.drawContours(cell1, [c], -1, 20, 1)
            imutils.showImage(cell1)
        digit = np.zeros_like(thresh)
        digit[y:y+h, x:x+w] = 255
        thresh[digit < 255] = 0
        return imutils.straightenAndCenter(thresh, (28, 28))
    if debug and not showDigitsOnly:
        imutils.showImage(thresh)
    return None


for imgname in os.listdir("images"):
    if imgname != "image30.jpg":
        continue
    img = imutils.resize(cv2.imread("images\\"+imgname), height=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cleaned = cv2.GaussianBlur(gray, (51, 51), 0)
    cleaned = cv2.divide(gray, cleaned, scale=255)
    thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (3, 3))
    minArea = 0.1*img.shape[0]*img.shape[1]
    if debug:
        imutils.showImage(thresh)
    cnts, _ = imutils.getExternalContours(img=thresh, original=True)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = None
    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * perimeter, True)
        if debug:
            imutils.displayBoundingBoxes(img, [cv2.boundingRect(c)])
        if len(approx) == 4:
            # img1 = img.copy()
            # cv2.drawContours(img1, [c], -1, (0, 255, 0), 2)
            # imutils.showImage(img1)
            x, y, w, h = cv2.boundingRect(approx)
            aspect = w * 1.0 / h
            if 0.8 > aspect > 1.25 or w*h < minArea:
                # print(aspect, w*h, minArea)
                continue
            cnt = approx
            break
    if cnt is None:
        print("Couldn't find box.")
        continue
        # raise Exception("Couldn't find puzzle box.")
    puzzle = imutils.four_point_transform(img, cnt.reshape(4, 2))
    warped = imutils.four_point_transform(cleaned, cnt.reshape(4, 2))
    imutils.showImage(puzzle)
    rows, cols = puzzle.shape[:2]
    window_height, window_width = rows//9, cols//9
    digits = []
    for i in range(9):
        for j in range(9):
            cell_ = warped[i*window_height:(i+1)*window_height, j*window_width:(j+1)*window_width]
            digit = extract_digit(cell_)
            if debug:
                if digit is not None:
                    imutils.showImage(digit)
                else:
                    print("No digit detected in this cell.")
            digits.append(digit)
    c = sum([1 for i in digits if i is not None])
    print(f"{c} digits detected.")
