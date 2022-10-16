import os
import shutil

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