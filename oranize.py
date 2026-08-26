import os
import glob
import datetime
import shutil


print("Program started on " + str(datetime.datetime.now()))


input_path = r"C:\Users\sanap\Downloads\archive (3)\train"
output_path = r"C:\Users\sanap\Downloads\output"

class_names = os.listdir(input_path)


for cla in class_names:


    inp = os.path.join(input_path, cla)

    if not os.path.isdir(inp):
        continue

    class_fini = os.path.join(output_path, cla)
    os.makedirs(class_fini, exist_ok=True)

    image_paths = glob.glob(os.path.join(inp, "*.jpg"))
  
    for image in image_paths:
        shutil.copy(image, class_fini)

    print(cla, "->", len(image_paths), "images copied")

# Print end time
print("Program finished on " + str(datetime.datetime.now()))
