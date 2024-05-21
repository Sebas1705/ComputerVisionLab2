import os
import Common.FileFuncs as ff
import numpy as np
from typing import List
from cv2.typing import MatLike
from Classes.Ex1AndEx2.ImagePreprocessor import ImagePreprocessor
from Settings import IMAGES_PATH

class ImageLoader:
    
    def __init__(
        self,
        origen_path: str,
        new_path: str,
        save: bool = False
    ) -> None:
        self.image: List[MatLike] = []
        self.origen_path: str = origen_path
        self.new_path: str = new_path
        self.save: bool = save
        
    def __get_all_directories(
        self,
        dir:str
    ) -> List[str]:
        """
        This function retrieves all directories within a given directory path.

        Parameters:
        -----------
        dir : str
            The directory path from which to retrieve subdirectories.

        Returns:
        -----------
        List[str]
            A list of all subdirectory paths within the given directory.

        Notes:
        -------
        This function first retrieves all subdirectories within the given directory.
        Then, it appends two extra subdirectories (dirs[-2] and dirs[-1]) to the list.
        Finally, it iterates over these extra subdirectories and retrieves all subdirectories within them.
        The function returns a list of all subdirectory paths.
        """
        
        dirs: List[str] = [dir+d for d in os.listdir(IMAGES_PATH+dir)]
        extras: List[str] = [dirs[-2],dirs[-1]]
        dirs = dirs[:-2]
        for ext in extras:
            for dir in os.listdir(IMAGES_PATH+ext):
                dirs.append(ext+"\\"+dir)
        return dirs

    def __charge_train_images(
        self,
        dirs:List[str]
    ) -> List[tuple[List[MatLike],str]]:
        """
        This function loads images from the given directories and returns a list of tuples.
        Each tuple contains a list of images and their corresponding directory path.

        Parameters:
        -----------
        dirs : List[str]
            A list of directory paths where the images are located.

        Returns:
        -----------
        List[tuple[List[MatLike],str]]
            A list of tuples. Each tuple contains a list of images and their corresponding directory path.
            The images are loaded using the `read_images` function from the `ff` module.

        Notes:
        -------
        This function iterates over the input list of directory paths, loads the images using the `read_images` function,
        and appends a tuple containing the loaded images and their corresponding directory path to the output list.
        Finally, it returns the output list.
        """
        
        images_dir: List[tuple[List[MatLike],str]] = []
        for dir in dirs:
            images_dir.append((ff.read_images(IMAGES_PATH+dir),dir))
        return images_dir

    def __create_images_preprocessors(
        self,
        images_dir: List[tuple[List[MatLike],str]]
    ) -> List[ImagePreprocessor]:
        """
        This function creates a list of ImagePreprocessor objects. Each ImagePreprocessor object is initialized with a tuple containing
        a list of images and their corresponding directory path.

        Parameters:
        -----------
        images_dir : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of images and their corresponding directory path.

        Returns:
        -----------
        List[ImagePreprocessor]
            A list of ImagePreprocessor objects, where each object is initialized with a tuple from the input list.

        Notes:
        -------
        This function iterates over the input list of tuples, creates an ImagePreprocessor object for each tuple,
        and appends the object to a list. Finally, it returns the list of ImagePreprocessor objects.
        """

        imgs_preps: List[ImagePreprocessor] = []
        for tp in images_dir:
            img_prep: ImagePreprocessor = ImagePreprocessor(tp[1],tp[0])
            imgs_preps.append(img_prep)
        return imgs_preps

    def __save_characters(
        self,
        characters: List[tuple[List[MatLike],str]],
        image_dir: str,
        new_path: str
    ) -> None:
        """
        This function saves the processed character images into separate directories for uppercase and lowercase characters.

        Parameters:
        -----------
        characters : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of character images and their corresponding directory path.
        image_dir : str
            The directory path where the original images are located.
        new_path : str
            The directory path where the processed images will be saved.

        Returns:
        -----------
        None
            This function does not return any value. It saves the processed character images into separate directories.

        Notes:
        -------
        The function creates two directories within the 'new_path' directory: 'mayus' and 'minus'.
        It then iterates over the list of character images, extracts the directory path from the tuple,
        and saves the character images into the corresponding directory ('mayus' or 'inus') based on the directory path.
        """
        
        os.mkdir(IMAGES_PATH+f"{new_path}mayus")
        os.mkdir(IMAGES_PATH+f"{new_path}minus")
        for chars,str in characters:
            path = IMAGES_PATH+new_path+str
            path = path.split(image_dir)
            path = path[0]+path[1]
            ff.save_images(chars,path)
            
    def __generate_arrays(
        self,
        characters: List[tuple[List[MatLike],str]]
    ) -> tuple[np.ndarray,np.ndarray]:
        """
        This function generates feature and label arrays from a list of character images.

        Parameters:
        -----------
        characters : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of character images and their corresponding directory path.

        Returns:
        -----------
        tuple[np.ndarray,np.ndarray]
            A tuple containing two numpy arrays: the first array represents the features of the characters,
            and the second array represents the corresponding labels.

        Notes:
        -------
        The function iterates over the list of character images, flattens each image into a 1D array,
        and appends it to the 'c' list along with its corresponding label (1-62).
        Finally, it converts the 'c' and 'e' lists into numpy arrays and returns them.
        """
        
        c = [] # Caracteristicas
        e = [] # Etiquetas
        i=1
        for chars,_ in characters:
            for char in chars:
                c.append(np.ravel(np.array(char)))
                e.append(i)
            i+=1
        return np.array(c),np.array(e)

    def get_arrays(
        self,
    ) -> tuple[np.ndarray,np.ndarray]:
        """
        This function is responsible for loading images, processing them, and generating arrays for training and testing.

        Parameters:
        -----------
        image_dir : str
            The directory path where the original images are located.
        new_path : str
            The directory path where the processed images will be saved.

        Returns:
        -----------
        tuple[np.ndarray,np.ndarray]
            A tuple containing two numpy arrays: the first array represents the features of the characters,
            and the second array represents the corresponding labels.
        """
        
        #Remover el contenido de las imagenes procesadas:
        if self.save:
            ff.remove_directory_content(IMAGES_PATH+self.new_path)
        
        #Cargar las imágenes:
        dirs: List[str] = self.__get_all_directories(self.origen_path)
        images_dir: List[tuple[List[MatLike],str]] = self.__charge_train_images(dirs)
        
        #Procesar las imagenes y obtener los caracteres:
        imgs_preps: List[ImagePreprocessor] = self.__create_images_preprocessors(images_dir)
        characters: List[tuple[List[MatLike],str]] = [
            (prep.proccess_images(),prep.path) 
            for prep in imgs_preps
        ]
        
        #Guardar las imágenes procesadas:
        if self.save:
            self.__save_characters(characters,self.origen_path,self.new_path)
        
        #Crear los arrays:
        return self.__generate_arrays(characters)