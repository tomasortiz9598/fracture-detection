import os
import shutil

path = "./dataset/"

dir = os.listdir(path)
for folder in dir:
  if os.listdir(path +folder)[0].endswith("negative"):
    print(path+folder + "   deleted")
    shutil.rmtree(path+folder)