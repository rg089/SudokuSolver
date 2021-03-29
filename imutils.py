import os, sys
import numpy as np
import cv2
import argparse
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from threading import Thread
import time
from queue import Queue

sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def get_args(dataset=False, save_model=False, save_plot=False, save_json=False, save_weights=False, single_image=False, pretrained_model=False, use_video=False, **kwargs):
    ap = argparse.ArgumentParser()
    if dataset:
        help_ = "path to input dataset"
        if "dataset" in kwargs:
            help_ = kwargs["dataset"]
        ap.add_argument("-d", "--dataset", required=True, help=help_)
    if save_model:
        help_ = "path to output model"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", type=str, default="inception", help=help_)
    elif pretrained_model:
        help_ = "name of pretrained model to use"
        if "model" in kwargs:
            help_ = kwargs["model"]
        ap.add_argument("-m", "--model", required=True, help=help_)
    if save_plot:
        help_ = "path to output plot"
        if "plot" in kwargs:
            help_ = kwargs["plot"]
        ap.add_argument("-p", "--plot", required=True, help=help_)
    if save_json:
        help_ = "path to output json"
        if "json" in kwargs:
            help_ = kwargs["json"]
        ap.add_argument("-j", "--json", required=True, help=help_)
    if save_weights:
        help_ = "path to weights directory"
        if "weights" in kwargs:
            help_ = kwargs["weights"]
        ap.add_argument("-w", "--weights", required=True, help=help_)
    if use_video:
        help_ = "path to (optional) video file"
        if "video" in kwargs:
            help_ = kwargs["video"]
        ap.add_argument("-v", "--video", help=help_)
    if single_image:
        help_ = "path to the input image"
        if "image" in kwargs:
            help_ = kwargs["image"]
        ap.add_argument("-i", "--image", help=help_)
    args = vars(ap.parse_args())
    return args


def normalize(*args):
    ans = []
    for i in args:
        ans.append(i.astype("float") / 255.0)
    if len(args) == 1:
        return ans[0]
    return ans


def encodeY(*args, ohe=True):
    if ohe:
        encoder = LabelBinarizer()
    else:
        encoder = LabelEncoder()
    ans = [encoder, encoder.fit_transform(args[0])]
    for i in range(1, len(args)):
        ans.append(encoder.transform(args[i]))
    return ans


def addDimension(*args, after=True):
    ans = []
    if after:
        for i in args:
            ans.append(i[..., np.newaxis])
    else:
        for i in args:
            ans.append(i[np.newaxis, ...])
    if len(args) == 1:
        return ans[0]
    return ans


def indexOfFirstString(l):
    for i in range(len(l)):
        if type(l[i]) == str:
            return i
    return len(l)


def showImage(*imgs, together=False, **kwnames):
    i = indexOfFirstString(imgs)
    images = imgs[:i]
    names = list(imgs[i:])
    if not together:
        if len(images) > len(names):
            if len(kwnames) != 0:
                names.extend(list(kwnames.values())[:len(images) - len(names)])
            names.extend([f"Image{i}" for i in range(len(names) + 1, len(images) + 1)])
        for i in range(len(images)):
            cv2.imshow(names[i], images[i])
    else:
        if len(names) != 0:
            name = names[0]
        elif "name" in kwnames:
            name = kwnames["name"]
        elif len(kwnames) != 0:
            name = list(kwnames.values())[0]
        else:
            name = "Image"
        cv2.imshow(name, np.hstack(images))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translate(im, x, y):
    """
    Translates image by specified amount.
    :param im: cv2 image object
    :param x: horizontal translation (-ve sign -> left)
    :param y: vertical translation (-ve sign -> up)
    :return: shifted cv2 image object
    """
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(im, M, (im.shape[1], im.shape[0]))


def rotate(im, angle, center=None, scale=1.0):
    """
    Rotates image by specified angle.
    :param im: cv2 image object
    :param angle: angle to rotate by
    :param center: point around which to rotate (default - center of image)
    :param scale: scale of the resultant image (default - same scale)
    :return: rotated image
    """
    h, w = im.shape[:2]
    if not center:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(im, M, (w, h))


def resize(img, width=None, height=None, inter=cv2.INTER_AREA, fx=1, fy=1, padSame=False):
    """
    Returns the resized image.
    :param img: cv2 image object
    :param width: The fixed width of the result
    :param height: The fixed height of the result
    :param inter: Interpolation Method
    :param fx: Ratio to scale width
    :param fy: Ratio to scale height
    :param padSame: If border is same or 0.
    :return: cv2 Image
    """
    h, w = img.shape[:2]

    if not width and not height:
        wn = int(w * fx)
        hn = int(h * fy)
        dim = (wn, hn)

    elif not width:
        r = height / float(h)
        dim = (int(w * r), height)

    elif not height:
        r = width / float(w)
        dim = (width, int(h * r))

    else:
        dW = 0; dH = 0
        if w < h:
            img = resize(img, width=width)
            dH = int((img.shape[0]-height)/2.0)
        else:
            img = resize(img, height=height)
            dW = int((img.shape[1] - width) / 2.0)
        h, w = img.shape[:2]
        img = img[dH:h-dH, dW:w-dW]
        dim = (width, height)

    return cv2.resize(img, dim, interpolation=inter)


def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    possibilities = ["left-to-right", "right-to-left", "top-to-bottom", "bottom-to-top"]
    assert method in possibilities
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == possibilities[2] or method == possibilities[3]:
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    cnts, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes), key=lambda x: (x[1][i], x[1][1]), reverse=reverse))
    return cnts, boundingBoxes


def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def plot_model(history, epochs, validation=True, accuracy=True, loss=True, title="Training Analytics"):
    plt.style.use("ggplot")
    plt.figure()
    if loss:
        plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_loss"], label="validation_loss")
    if accuracy:
        plt.plot(np.arange(0, epochs), history.history["accuracy"], label="accuracy")
        if validation:
            plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="validation_accuracy")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Metric Values")
    plt.legend()
    return plt


def deskew_digit(img, applyThreshold=True):
    if applyThreshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    (h, w) = img.shape[:2]
    moments = cv2.moments(img)
    if abs(moments['mu02']) < 1e-2:
        return img
    skew = moments["mu11"] / moments["mu02"]
    M = np.float32([
        [1, skew, -0.5 * w * skew],
        [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def deskew_text(img, applyThreshold=True):
    if applyThreshold:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = img
    cords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(cords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def pointInside(r1, r2, leverage):
    """
    Checks whether the rectangle r2 is inside r1.
    :param r1: tuple of the form (x,y,w,h)
    :param r2: tuple of the form (x,y,w,h)
    :return: True if inside else False
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    if x1 - leverage <= x2 <= x1 + w1 + leverage and y1 - leverage <= y2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 + w2 <= x1 + w1 + leverage and y1 - leverage <= y2 + h2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 <= x1 + w1 + leverage and y1 - leverage <= y2 + h2 <= y1 + h1 + leverage:
        return True
    elif x1 - leverage <= x2 + w2 <= x1 + w1 + leverage and y1 - leverage <= y2 <= y1 + h1 + leverage:
        return True
    return False


def mergeRects(r1, r2):
    """
    merge 2 rectangles r1 and r2 where r1 has lower x co-ordinate.
    :param r1: first rectangle
    :param r2: second rectangle
    :return: merged rectangle
    """
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x3 = min(x1, x2)
    y3 = min(y1, y2)
    w3 = max(x1 + w1, x2 + w2) - x3
    h3 = max(y1 + h1, y2 + h2) - y3
    return x3, y3, w3, h3


def getExternalContours(img=None, cnts=None, leverage=3, applySort=False, minArea=None, display=False, original=False):
    col = (0, 255, 0)
    if img is not None and cnts is None:
        col = 255
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, boundingBoxes = sort_contours(cnts)
    else:
        if not cnts:
            raise AssertionError("Please provide either the image or the contours.")
        if applySort:
            cnts, boundingBoxes = sort_contours(cnts)
        else:
            boundingBoxes = [cv2.boundingRect(cnt) for cnt in cnts]
    # Assuming the contours are sorted left to right and top ro bottom so the parent contour comes before its children.
    if original:
        return cnts, boundingBoxes
    contours = [cnts[0]]
    boundingBoxes1 = [boundingBoxes[0]]
    for i in range(1, len(boundingBoxes)):
        x, y, w, h = boundingBoxes[i]
        if minArea and w * h < minArea:
            continue
        if not pointInside(boundingBoxes1[-1], boundingBoxes[i], leverage):
            contours.append(cnts[i])
            boundingBoxes1.append(boundingBoxes[i])
        else:
            boundingBoxes1[-1] = mergeRects(boundingBoxes1[-1], boundingBoxes[i])
    if display:
        if img is None:
            print("No image given to display.")
        else:
            displayBoundingBoxes(img, boundingBoxes1, col)
            displayBoundingBoxes(img, boundingBoxes, col)
    return contours, boundingBoxes1


def displayBoundingBoxes(img, boundingBoxes, col=(0, 255, 0)):
    img1 = img.copy()
    for b in boundingBoxes:
        x, y, w, h = b
        cv2.rectangle(img1, (x, y), (x + w, y + h), col, 1)
    showImage(img1)


def centralize_digit(img, size):
    (eW, eH) = size
    if img.shape[1] > img.shape[0]:
        image = resize(img, width=eW, inter=cv2.INTER_AREA)
    else:
        image = resize(img, height=eH, inter=cv2.INTER_AREA)
    extent = np.zeros((eH, eW), dtype="uint8")
    offsetX = (eW - image.shape[1]) // 2
    offsetY = (eH - image.shape[0]) // 2
    extent[offsetY:offsetY + image.shape[0], offsetX:offsetX + image.shape[1]] = image
    M = cv2.moments(extent)
    try:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        (dX, dY) = ((size[0] // 2) - cX, (size[1] // 2) - cY)
        M = np.float32([[1, 0, dX], [0, 1, dY]])
        extent = cv2.warpAffine(extent, M, size)
        return extent
    except ZeroDivisionError:
        return resize(img, width=size[0], height=size[1])


def deskew_digits(imgs, applyThreshold=False):
    return np.array([deskew_digit(img, applyThreshold) for img in imgs])


def centralize_digits(imgs, size):
    return np.array([centralize_digit(img, size) for img in imgs])


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-co-ordinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x co-ordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def straightenAndCenter(img, size, applyThreshold=False):
    img = deskew_digit(img, applyThreshold)
    img = centralize_digit(img, size)
    return img


class FileVideoStream:
    def __init__(self, path, transform=None, queue_size=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.transform = transform

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queue_size)
        # initialize thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        # start a thread to read frames from the file video stream
        self.thread.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                break

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stopped = True

                # if there are transforms to be done, might as well
                # do them on producer thread before handing back to
                # consumer thread. ie. Usually the producer is so far
                # ahead of consumer that we have time to spare.
                #
                # Python is not parallel but the transform operations
                # are usually OpenCV native so release the GIL.
                #
                # Really just trying to avoid spinning up additional
                # native threads and overheads of additional
                # producer/consumer queues since this one was generally
                # idle grabbing frames.
                if self.transform:
                    frame = self.transform(frame)

                # add the frame to the queue
                self.Q.put(frame)
            else:
                time.sleep(0.1)  # Rest for 10ms, we have a full queue

        self.stream.release()

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    # Insufficient to have consumer use while(more()) which does
    # not take into account if the producer has reached end of
    # file stream.
    def running(self):
        return self.more() or not self.stopped

    def more(self):
        # return True if there are still frames in the queue. If stream is not stopped, try to wait a moment
        tries = 0
        while self.Q.qsize() == 0 and not self.stopped and tries < 5:
            time.sleep(0.1)
            tries += 1

        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        # wait until stream resources are released (producer thread might be still grabbing frame)
        self.thread.join()