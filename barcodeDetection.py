import numpy as np
import cv2

def detect(filename):
    # open the image
    image = cv2.imread(filename)

    #grey scale it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #compute scharr gradient mag representation of the image
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y gradient from the x gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # blur the image  with size 9 as to eliminate false positives
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # erode small whites spots and dialate the surviving white spaces
    # isolates the barcode area
    closed = cv2.erode(closed, None, iterations=7)
    closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thrshold image
    img, cnts, hierarchy = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if there is no cnts then there are no barcodes in this image
    if len(cnts) == 0:
        return None

    # sort the contours by area and take the largest
    # then compute a bounding box of the largest
    a = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rectangle = cv2.minAreaRect(a)
    box = np.int0(cv2.boxPoints(rectangle))

    # extract the points needed to crop image
    point = find_bound(box)

    print("Bounding points: \nLeft: ", point[0], "\nRight: ", point[1], "\nTop: ", point[2], "\nBottom: ", point[3])

    # crop the image down to the specified bounding box
    crop = image[point[2]-20:point[3]+20, point[0]-20:point[1]+20]
    #$cv2.imshow("Cropped", crop)
    # cv2.drawContours(image, [box], -1, (255, 0, 0), 3)
    # cv2.imshow("Cropped", image)
    #cv2.waitKey(0)

    # return the cropped image
    return crop


# takes the points of a bounding box andd returns an array of [left, right, top, bottom]
def find_bound(points):
    one = []
    two = []
    for i in points:
        one.append(i[0])
        two.append(i[1])

    xs = min(int(j) for j in one)
    xl = max(int(k) for k in one)
    ys = min(int(l) for l in two)
    yl = max(int(m) for m in two)

    return [xs, xl, ys, yl]


if __name__ == '__main__':
    image = input("-enter image: ")
    result = detect(image)
    print("Received cropped image!")
    cv2.imshow("Result", result)
    cv2.waitKey(0)
