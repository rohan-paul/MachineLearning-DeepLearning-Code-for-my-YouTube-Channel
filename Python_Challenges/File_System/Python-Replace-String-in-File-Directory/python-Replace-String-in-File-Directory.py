import os

directory = '/home/paul/Pictures'

[os.rename(os.path.join(directory, f), os.path.join(directory, f).replace('some-string', 'other-string')) for f in os.listdir(directory)]

