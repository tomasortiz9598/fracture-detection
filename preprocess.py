import os
import shutil
import cv2
def move_files():
  path = "./dataset/"

  dir = os.listdir(path)
  print(len(dir))

  for folder in dir:
    print(folder)
    if os.listdir(path + folder)[0].endswith("positive"):
      org = path + folder + "/study1_positive/"
      if len(os.listdir(org))>2:
        org += "image3.png"
      dst = path + folder.replace("patient", "") +".png"
      print(org + "  moved to " + dst)
      shutil.move(org, dst)


def roı_clahe_pre_process(folder, new_folder):
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)  # read image from directory
    gray = img
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert rgb to gray scala for apply threshold
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # apply binary and otsu threshold
    x, y, w, h = cv2.boundingRect(thresh)  # determine boundary of ROI
    x, y, w, h = x, y, w + 20, h + 20  # +20 pixels tolerans for boundaries
    img = img[y:y + h, x:x + w]  # crop original image with the help of boundary

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # determine clahe values
    img = clahe.apply(img)  # apply clahe transform on image

    new_path = new_folder + filename  # determine new path for save to image
    cv2.imwrite(new_path, img)  # save output to new paths


roı_clahe_pre_process("./dataset/compuestas", "./modified_ds/compuestas/")
