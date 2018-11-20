import barcode
from barcode.writer import ImageWriter
import array
import random
import csv

nameList = list()
charArray = array.array('u', ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                              'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                              '-', '.', ' ', '$', '/', '+', '%'])


def barcode_value_gen(name_len):
    while True:
        temp_name = ''
        for j in range(name_len):
            temp_name = temp_name + charArray[random.randint(0, 42)]

        if temp_name not in nameList:
            nameList.append(temp_name)
            # print("Generated value: ", temp_name)
            break
    return temp_name


def barcode_img_gen(barcode_value):
    code_39 = barcode.get_barcode_class('code39')
    return code_39(barcode_value, writer=ImageWriter())


if __name__ == '__main__':
    counter = 1
    x = 8
    with open('./barcode_images/barcode_spreadsheet.csv', 'w') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(['File_Name', 'Barcode_Value'])
        x = x - 1
        for y in range(43):
            temp_val = barcode_value_gen(x)
            temp_barcode_image = barcode_img_gen(temp_val)

            # print('Generated barcode: ' + temp_barcode_image.get_fullcode())
            file_name = './barcode_images/code39_' + repr(counter)
            file = temp_barcode_image.save(file_name)
            file_writer.writerow([file_name, temp_barcode_image.get_fullcode()])

            print('File ' + repr(counter) + ' Generated')
            counter += 1
            if counter == 5:
                exit(0)

    print('Finished Generating All Files!')
