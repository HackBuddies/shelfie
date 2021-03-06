import pyocr
import pyocr.builders
import sys
from PIL import Image
import time
import cv2

TEMP_DIR = "tess_temp/"


def closest(last):
    first = last[0]
    end = "9"
    if first != "4":
        first = "9"
    return first + end


def get_price(img):
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    temp_fname = TEMP_DIR + str(time.time()) + ".jpg"
    cv2.imwrite(temp_fname, img)
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    # print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    # print("Available languages: %s" % ", ".join(langs))
    lang = "eng"
    # print("Will use lang '%s'" % (lang))
    # Ex: Will use lang 'fra'
    # Note that languages are NOT sorted in any way. Please refer
    # to the system locale settings for the default language
    # to use.
    time.sleep(1)
    digits = tool.image_to_string(
        Image.open(temp_fname),
        lang=lang,
        builder=pyocr.tesseract.DigitBuilder()
    )

    digits = digits.replace(" ", "")[-4:]
    digits = digits.replace("\n", "")[-4:]
    digits = digits.replace("\t", "")[-4:]
    if len(digits) < 2:
        digits = "1299"
    digits = digits[:2] + closest(digits[2:])
    digits = int(str(digits))/100.0
    return digits


def main():
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("No OCR tool found")
        sys.exit(1)
    # The tools are returned in the recommended order of usage
    tool = tools[0]
    print("Will use tool '%s'" % (tool.get_name()))
    # Ex: Will use tool 'libtesseract'

    langs = tool.get_available_languages()
    print("Available languages: %s" % ", ".join(langs))
    lang = "eng"
    print("Will use lang '%s'" % (lang))
    # Ex: Will use lang 'fra'
    # Note that languages are NOT sorted in any way. Please refer
    # to the system locale settings for the default language
    # to use.


    img_fname = "labels/019.jpg"

    txt = tool.image_to_string(
        Image.open(img_fname),
        lang=lang,
        builder=pyocr.builders.TextBuilder()
    )

    print(txt)
    # txt is a Python string

    word_boxes = tool.image_to_string(
        Image.open(img_fname),
        lang="eng",
        builder=pyocr.builders.WordBoxBuilder()
    )
    # list of box objects. For each box object:
    #   box.content is the word in the box
    #   box.position is its position on the page (in pixels)
    #
    # Beware that some OCR tools (Tesseract for instance)
    # # may return empty boxes
    #
    # line_and_word_boxes = tool.image_to_string(
    #     Image.open(img_fname), lang="eng",
    #     builder=pyocr.builders.LineBoxBuilder()
    # )
    # list of line objects. For each line object:
    #   line.word_boxes is a list of word boxes (the individual words in the line)
    #   line.content is the whole text of the line
    #   line.position is the position of the whole line on the page (in pixels)
    #
    # Beware that some OCR tools (Tesseract for instance)
    # may return empty boxes

    # # Digits - Only Tesseract (not 'libtesseract' yet !)
    # digits = tool.image_to_string(
    #     Image.open(img_fname),
    #     lang=lang,
    #     builder=pyocr.tesseract.DigitBuilder()
    # )

    print(digits)
    # digits is a python string
