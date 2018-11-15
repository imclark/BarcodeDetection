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
    left  = box[0][0]
    right = box[2][0]
    top = box[1][1]
    bottom = box[0][1]

    print("Bounding points: \nLeft: ", left, "\nRight: ", right, "\nTop: ", top, "\nBottom: ", bottom)

    # crop the image down to the specified bounding box
    crop = image[top:bottom, left:right]
    cv2.imshow("Cropped", crop)
    # cv2.drawContours(image, [box], -1, (255, 0, 0), 3)
    # cv2.imshow("Cropped", image)
    cv2.waitKey(0)

    # return the cropped image
    return crop


if __name__ == '__main__':
    image = input("-enter image: ")
    result = detect(image)
    print("Received cropped image!")
    cv2.imshow("Result", result)
    cv2.waitKey(0)

