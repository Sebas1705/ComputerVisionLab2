from numpy import ndarray
from Classes.Ex1.ImagePreprocessor import ImagePreprocessor
from settings import IMAGES_PATH,SAVE_PROCESED_IMAGES
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
    characters: List[tuple[List[MatLike],str]],
    image_dir: str,
    new_path: str
) -> None:
    os.mkdir(IMAGES_PATH+f"{new_path}mayus")
    os.mkdir(IMAGES_PATH+f"{new_path}minus")
    for chars,str in characters:
        path=IMAGES_PATH+new_path+str
        path = path.split(image_dir)
        path = path[0]+path[1]
        ff.save_images(chars,path)
        
def __generate_arrays(
    characters: List[tuple[List[MatLike],str]]
) -> tuple[ndarray,ndarray]:
    x: ndarray = []
    y: ndarray = []
    for chars,str in characters:
        x.append(chars)
        y.append(str)
    return x,y

def __get_arrays(
    image_dir: str,
    new_path: str
) -> tuple[ndarray,ndarray]:
    
    #Remover el contenido de las imagenes procesadas:
    ff.remove_directory_content(IMAGES_PATH+new_path)
    
    #Cargar las imágenes:
    dirs: List[str] = __get_all_directories(image_dir)
    images_dir: List[tuple[List[MatLike],str]] = __charge_train_images(dirs)
    
    #Procesar las imagenes y obtener los caracteres:
    imgs_preps: List[ImagePreprocessor] = __create_images_preprocessors(images_dir)
    characters: List[tuple[List[MatLike],str]] = [
        (prep.proccess_images(),prep.path) 
        for prep in imgs_preps
    ]
    
    #Guardar las imágenes procesadas:
    if SAVE_PROCESED_IMAGES:
        __save_characters(characters,image_dir,new_path)
    
    #Crear los arrays:
    return __generate_arrays(characters)
    
def exec1() -> None:
    
    #Train arrays:
    x_train,y_train = __get_arrays("train_ocr_origen\\","train_ocr_new\\")
    
    #Test arrays:
    x_test,y_test = __get_arrays("validation_ocr_origen\\","validation_ocr_new\\")
    
    
    
    
    
    
    
    
    
