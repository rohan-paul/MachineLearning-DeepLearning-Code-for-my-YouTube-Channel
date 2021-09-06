# Youtube Link => https://youtu.be/XI1FA1dehxk

import os
from os import listdir

folder_path = "/home/paul/Documents/"


def get_cleaned_filename(pathname):
    return "".join(f for f in pathname if f.isalpha())


for filename in listdir(folder_path):
    src = folder_path + filename
    dst = folder_path + get_cleaned_filename(filename)

    os.rename(src, dst)
