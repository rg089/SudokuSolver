import cv2
import numpy as np
import imutils


class Preprocessor:
    def __init__(self, width, height, debug=False, showDigitsOnly=True):
        self.width = width
        self.height = height
        self.debug = debug
        self.showDigitsOnly = showDigitsOnly

    def extract_digit(self, cell):
        """
        Extracts digit and returns a binary image with the digit straightened and centered. If no digit found, returns None.
        :param cell: The cleaned grayscale image of the cell.
        :return: cv2 image object of the digit (binary), else None if empty cell.
        """
        thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        height, width = thresh.shape[:2]
        minArea = 0.05 * width * height
        maxArea = 0.9 * width * height

        roi = thresh[int(0.3 * height): int(0.7 * height), int(0.3 * width):int(0.7 * height)]  # Define the center region as an area where some pixels of the digit must occur.
        cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(cnts) == 0 or (roi > 0).sum() / (roi.shape[0] * roi.shape[1]) < 0.02:  # No contours found or less than 2% white pixels in the center area
            if self.debug and not self.showDigitsOnly:
                thresh1 = thresh.copy()
                cv2.rectangle(thresh1, (int(0.3 * width), int(0.3 * height)), (int(0.7 * width), int(0.7 * height)), 255, 1)
                imutils.showImage(thresh1)
            return None

        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]  # Get the three max contours according to their area.
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            boundBox = thresh[y:y + h, x:x + w]  # Get the bounding box of the contour.
            if x == 0 or x + w == width or w < 3 or h < 3 or w >= width-2 or h >= height-2 or w * h < minArea or (boundBox > 0).sum() / (width * height) < 0.0225 or w * h > maxArea:
                # If the contour starts or ends at the extreme left or right, has extreme (low or high) width or height or
                # area less than 5% of the image, or more than 90% of the image, or is less than 2.25% filled, then it is an empty cell.
                if self.debug and not self.showDigitsOnly:
                    print("No digits detected. Details:", x, y, w, h, (boundBox > 0).sum() / (width * height) * 100)
                continue

            if self.debug and not self.showDigitsOnly:
                cell1 = cell.copy()
                print(x, y, w, h)
                cv2.drawContours(cell1, [c], -1, 20, 1)
                imutils.showImage(cell1)

            mask = np.zeros_like(thresh)
            mask[y:y + h, x:x + w] = 255  # Create a mask for the bounding box and color it where digit is found.
            thresh[mask < 255] = 0  # Color everything except mask as black in thresh.
            ans = imutils.straightenAndCenter(thresh, (self.width, self.height))  # Deskew and centralize digit and return it.
            if self.debug:
                imutils.showImage(ans)
            return ans

        # No digits found!
        if self.debug and not self.showDigitsOnly:
            imutils.showImage(thresh)
        return None  # Return None.

    def extract_grid(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        cleaned = cv2.GaussianBlur(gray, (51, 51), 0)
        cleaned = cv2.divide(gray, cleaned, scale=255)  # Clean the image

        thresh = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, (3, 3))  # Close any small gaps

        minArea = 0.1 * img.shape[0] * img.shape[1]  # Area of the grid should be greater than 10% of the image.

        if self.debug and not self.showDigitsOnly:
            imutils.showImage(thresh)

        cnts, _ = imutils.getExternalContours(img=thresh, original=True)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # Find and sort contours by area.

        cnt = None
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * perimeter, True)
            if len(approx) == 4:  # Detect shapes that look like rectangles.
                x, y, w, h = cv2.boundingRect(approx)
                aspect = w * 1.0 / h
                if 0.8 > aspect > 1.25 or w * h < minArea:  # If width is much more than height or vice versa, or the area is very small, then don't chose that.
                    continue
                cnt = approx  # This is the grid.
                break

        if cnt is None:
            # print("Couldn't find puzzle box.")
            return None, None, None, None, None

        keypts_img = cnt.reshape(4, 2)
        sudoku = imutils.four_point_transform(img, keypts_img)  # Perspective transform the grid to get a straight view.
        sudoku_gray = imutils.four_point_transform(cleaned, keypts_img)

        if (keypts_img[0][0]) ** 2 + (keypts_img[0][1]) ** 2 < (keypts_img[1][0]) ** 2 + (keypts_img[1][1]) ** 2:
            keypts_grid = np.array([[0, 0], [0, sudoku.shape[0]], [sudoku.shape[1], sudoku.shape[0]], [sudoku.shape[1], 0]])
        else:
            keypts_grid = np.array([[sudoku.shape[1], 0], [0, 0], [0, sudoku.shape[0]], [sudoku.shape[1], sudoku.shape[0]]])

        if self.debug and not self.showDigitsOnly:
            imutils.showImage(sudoku)

        new_img = img.copy()
        cv2.drawContours(new_img, [cnt], -1, (0, 0, 0), -1)

        return sudoku, sudoku_gray, keypts_img, keypts_grid, new_img
