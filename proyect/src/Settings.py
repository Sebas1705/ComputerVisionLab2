import os

#---Flags:
DEBUG_MODE = True
NEW_CRS = False

#---FileFuncs:
GLOBAL_PATH = os.path.abspath(__file__).replace("proyect\\src\\Settings.py","")
IMAGES_PATH = GLOBAL_PATH + "proyect\\images\\"
FILES_PATH = GLOBAL_PATH + "proyect\\files\\"
PKL_PATH = FILES_PATH + "Pkls\\"
