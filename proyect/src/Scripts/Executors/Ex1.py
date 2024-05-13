from Classes.Ex1.ImagePreprocessor import ImagePreprocessor
from settings import IMAGES_PATH
import Common.FileFuncs as ff
from typing import List
from cv2.typing import MatLike
import os

def __get_all_directories(dir:str) -> List[str]:
    dirs: List[str] = [dir+d for d in os.listdir(IMAGES_PATH+dir)]
    extras: List[str] = [dirs[-2],dirs[-1]]
    dirs = dirs[:-2]
    for ext in extras:
        for dir in os.listdir(IMAGES_PATH+ext):
            dirs.append(ext+"\\"+dir)
    return dirs

def __charge_train_images(dirs:List[str]) -> List[tuple[List[MatLike],str]]:
    images_dir: List[tuple[List[MatLike],str]] = []
    for dir in dirs:
        images_dir.append((ff.read_images(IMAGES_PATH+dir),dir))
    return images_dir

def __create_images_preprocessors(
    images_dir: List[tuple[List[MatLike],str]]
) -> List[ImagePreprocessor]:
    imgs_preps: List[ImagePreprocessor] = []
    for tp in images_dir:
        img_prep: ImagePreprocessor = ImagePreprocessor(tp[1],tp[0])
        imgs_preps.append(img_prep)
    return imgs_preps

def __save_characters(
    characters: List[tuple[List[MatLike],str]]
) -> None:
    os.mkdir(IMAGES_PATH+"train_ocr_new\\mayus")
    os.mkdir(IMAGES_PATH+"train_ocr_new\\minus")
    for chars,str in characters:
        path=IMAGES_PATH+"train_ocr_new\\"+str
        path = path.split("train_ocr_origen\\")
        path = path[0]+path[1]
        ff.save_images(chars,path)
    
def exec1() -> None:
    
    #Remover el contenido de las imagenes procesadas:
    ff.remove_directory_content(IMAGES_PATH+"train_ocr_new\\")
    
    #Cargar las imágenes:
    dirs: List[str] = __get_all_directories("train_ocr_origen\\")
    images_dir: List[tuple[List[MatLike],str]] = __charge_train_images(dirs)
    
    #Procesar las imagenes y obtener los caracteres:
    imgs_preps: List[ImagePreprocessor] = __create_images_preprocessors(images_dir)
    characters: List[tuple[List[MatLike],str]] = [
        (prep.proccess_images(),prep.path) 
        for prep in imgs_preps
    ]
    
    #Guardar las imágenes procesadas:
    __save_characters(characters)
    
    
    
    
