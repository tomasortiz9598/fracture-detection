import os
import shutil

path = "./dataset/"

dir = os.listdir(path)
print(len(dir))
for folder in dir:
  pass
  if os.listdir(path + folder)[0].endswith("positive"):
    org = path + folder + "/study1_positive/"
    if len(os.listdir(org))>1:
      org += "image2.png"
    dst = path + folder.replace("patient", "") +".png"
    print(org + "  moved to " + dst)
    shutil.move(org, dst)

