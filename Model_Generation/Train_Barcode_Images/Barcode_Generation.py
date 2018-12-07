import barcode
from barcode.writer import ImageWriter
import array
import random
import csv
import cv2
import numpy as np
import sys

nameList = list()
charArray = array.array('u', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'])
barcode_image_num = 20000


def barcode_value_gen(name_len):
    """
    Generates the value for the barcode

    :param: name_len: the length of the barcode value
    :return: he string of the unique barcode value
    """
    while True:
        temp_name = ''
        for j in range(name_len):
            temp_name = temp_name + charArray[random.randint(0, 35)]
        if temp_name not in nameList:
            nameList.append(temp_name)
            break
    return temp_name


def barcode_img_gen(barcode_value):
    """
    Generates the barcode image

    :param barcode_value: the string of the barcode value
    :return: the newly created barcode image
    """
    code_39 = barcode.get_barcode_class('code39')
    return code_39(barcode_value, writer=ImageWriter(), add_checksum=False)


def brightness_mod(image_file, brightness):
    """
    Modifies the brightness of the image

    :param image_file: the image to be modified
    :param brightness: the degree of brightness to adjust the image
    :return: returns the modified image
    """
    if brightness != 0:
        if brightness > 0:
            shade = brightness
            highlight = 255
        else:
            shade = 0
            highlight = 255 + brightness
        a = (highlight - shade) / 255
        b = shade

        temp_img = cv2.addWeighted(image_file, a, image_file, 0, b)
    else:
        temp_img = image_file.copy()
    return temp_img


def rotate(image_file, rotation):
    """
    Rotates the image by the specified degree

    :param image_file: the image to modify
    :param rotation: the degree to rotate the image
    :return: the rotated image
    """
    rows, cols, other = image_file.shape
    m = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    return cv2.warpAffine(image_file, m, (cols, rows))


def translate(image_file, x, y):
    """
    translate (shift objects locating) a given image

    :param image_file: the image to translate
    :param x: pixels to move image in the x direction
    :param y: pixels to move in the y direction
    :return: modified image
    """
    rows, cols, other = image_file.shape
    matrix = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(image_file, matrix, (cols, rows))


def image_mod(filename, count):
    """
    base method for modifying the training images

    :param filename: the file path for the image to be modded
    :param count: the number of the image
    :return: the modded image
    """
    img = cv2.imread(filename + '.png')

    if count / barcode_image_num >= 0.5:
        img = rotate(img, 180)

    if count % 5 == 0:
        if (count / 5) % 2 == 0:
            img = rotate(img, 2)
        else:
            img = rotate(img, -2)
        if count % 3 == 0:
            img = translate(img, 5, 5)
        else:
            img = translate(img, -5, -5)

    if count / barcode_image_num <= 0.3:
        img = brightness_mod(img, -127)

    if count / barcode_image_num >= 0.7:
        img = brightness_mod(img, 127)

    if count % 2 == 0:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    cv2.imwrite(filename + '.png', img)


if __name__ == '__main__':
    counter = 0
    with open('./barcode_images/csv/barcode_spreadsheet.csv', 'w') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['File_Name', 'Barcode_Value'])

        for z in range(barcode_image_num):
            temp_val = barcode_value_gen(13)
            temp_barcode_image = barcode_img_gen(temp_val)

            file_name = './barcode_images/code39_' + repr(counter)
            file = temp_barcode_image.save(file_name)
            file_writer.writerow(['code39_' + repr(counter), temp_barcode_image.get_fullcode()])

            image_mod(file_name, z)

            sys.stdout.write("\rCreated image %i / " % z + "%i" % barcode_image_num)
            sys.stdout.flush()
            counter += 1

    print('\n Finished Generating All Files!')
