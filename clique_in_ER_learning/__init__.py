import os
import sys

sys.path.append("..")
for directory in os.listdir(".."):
    sys.path.append(os.path.join("..", directory))
