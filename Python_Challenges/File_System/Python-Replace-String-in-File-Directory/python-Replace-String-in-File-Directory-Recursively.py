import os
import shutil
import sys

for root, dirs, files in os.walk(sys.argv[1], topdown=False):
    for f in files:
        shutil.move(os.path.join(root, f), os.path.join(root, f.replace('some-string', '').strip()))

    for dr in dirs:
        shutil.move(os.path.join(root, dr), os.path.join(root, dr.replace('some-string', '').strip()))
