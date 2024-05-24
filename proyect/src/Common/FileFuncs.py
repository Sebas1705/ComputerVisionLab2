from cv2.typing import MatLike
from Settings import IMAGES_PATH
from typing import List
import shutil
import cv2
import os

def save_images(
    images_to_save: List[MatLike],
    path:str,
    extra:str="",
    cv2Const:int=None
) -> None:
    """
    Saves a list of images to the specified path.

    Parameters:
    -----------
    images_to_save : List[MatLike] 
        A list of images to be saved.
    path : str
        The path to which the images are to be saved.
    extra : str, optional 
        An extra string to be appended to the file name. Defaults to "".
    cv2Const : int, optional
        An constant that transform an image into a colar format, Defaulst to None
    """
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(len(images_to_save)):
        img: MatLike = cv2.cvtColor(images_to_save[i],cv2Const) if cv2Const!=None else images_to_save[i]
        cv2.imwrite(f"{path}/{extra}{i:0>4}.png",img)
        
def read_images(
    path:str,
    cv2Const:int=cv2.COLOR_BGR2RGB
) -> list[MatLike]:
    """
    Reads a list of images from the specified path.

    Parameters:
    -----------
    path : str, optional
        The path from which the images are to be read.
        Defaults to "../../images/test".
    cv2Const : int, optional
        The constant used for converting the images.
        Defaults to cv2.COLOR_BGR2RGB.

    Returns:
    --------
    list[MatLike]
        A list of images read from the specified path.
    """
    return [
        cv2.cvtColor(cv2.imread(os.path.join(path,file)),cv2Const) 
        if cv2Const is not None else cv2.imread(os.path.join(path,file))
        for file in os.listdir(path)
        if file.endswith(".png") or file.endswith(".jpg")
    ]

def remove_directory_content(
    path:str
) -> None:
    """
    Removes the content of a directory.

    Parameters:
    -----------
    path : str
        The path of the directory to be emptied.
    """
    for file in os.listdir(path):
        path_complete = os.path.join(path,file)
        if os.path.isfile(path_complete):
            os.remove(path_complete)
        elif os.path.isdir(path_complete):
            shutil.rmtree(path_complete)

def create_txt(path:str,text:str):    
    """
    Creates a text file with the given text.

    Parameters:
    -----------
    path : str
        The path of the file to be created.
    text : str
        The text to be written in the file.
    """
    if os.path.exists(path):
        os.remove(path)    
    with open(path, 'w') as file:
        file.write(text)