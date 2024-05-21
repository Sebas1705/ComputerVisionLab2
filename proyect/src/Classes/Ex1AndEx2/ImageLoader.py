from numpy import ndarray
from typing import List
from cv2.typing import MatLike
from Classes.Ex1AndEx2.ImagePreprocessor import ImagePreprocessor
from Settings import IMAGES_PATH
import os
import Common.FileFuncs as ff
import numpy as np
class ImageLoader:
    
    def __init__(
        self,
        origen_path: str,
        new_path: str,
        save: bool = False
    ) -> None:
        """
        Initializes an instance of the ImageLoader class.

        Parameters:
        -----------
        origen_path : str
            The directory path where the original images are located.
        new_path : str
            The directory path where the processed images will be saved.
        save : bool, optional
            A flag indicating whether to save the processed images. Default is False.

        Returns:
        -----------
            None
        """
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
        This function first retrieves all subdirectories within the given directory path.
        Then, it appends two extra subdirectories from the parent directory to the list.
        Finally, it iterates over these extra subdirectories and retrieves their subdirectories,
        appending them to the list.
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
            A list of directory paths from which to load images.

        Returns:
        -----------
        List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of images and their corresponding directory path.
            The images are loaded using the read_images function from the Common.FileFuncs module.

        Notes:
        -------
        This function iterates over the input list of directory paths, loads the images from each directory using the
        read_images function, and appends a tuple containing the loaded images and their corresponding directory path
        to a list. Finally, it returns the list of tuples.
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
            The first element of the tuple is a list of images, and the second element is a string representing the directory path.

        Returns:
        -----------
        List[ImagePreprocessor]
            A list of ImagePreprocessor objects, where each object is initialized with a tuple from the input list.
            The ImagePreprocessor objects are responsible for processing the images and extracting character images.

        Notes:
        -------
        This function iterates over the input list of tuples, creates an ImagePreprocessor object for each tuple,
        and appends the object to a list. Finally, it returns the list of ImagePreprocessor objects.

        The ImagePreprocessor class is assumed to have a constructor that takes two parameters:
        - directory path (str)
        - list of images (List[MatLike])

        The ImagePreprocessor class is responsible for processing the images and extracting character images.
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
        This function saves the processed character images to the specified directory.

        Parameters:
        -----------
        characters : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of character images and their corresponding directory path.
        image_dir : str
            The original directory path where the images were loaded from.
        new_path : str
            The directory path where the processed images will be saved.

        Returns:
        -----------
            None

        Notes:
        -------
        This function first creates two directories within the new_path: 'ayus' and 'inus'.
        Then, it iterates over the list of character images, extracts the directory path from each tuple,
        and constructs a new path by replacing the image_dir with the new_path.
        Finally, it saves the character images to the new path using the save_images function from the Common.FileFuncs module.
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
    ) -> tuple[ndarray,ndarray]:
        """
        This function generates numpy arrays from the processed character images.

        Parameters:
        -----------
        characters : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of character images and their corresponding directory path.

        Returns:
        -----------
        tuple[ndarray,ndarray]
            A tuple containing two numpy arrays: the first array represents the features of the characters,
            and the second array represents the corresponding labels.

        Notes:
        -------
        This function iterates over the list of character images, extracts the features (pixel values) of each character,
        and appends them to the 'c' list. It also assigns a label to each character based on the directory path,
        and appends the label to the 'e' list. Finally, it converts the 'c' and 'e' lists to numpy arrays and returns them.
        """
        
        c = []
        e = [] 
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
        This method loads images, processes them, and generates numpy arrays for machine learning.

        Parameters:
        -----------
        None

        Returns:
        -----------
        tuple[np.ndarray,np.ndarray]
            A tuple containing two numpy arrays: the first array represents the features of the characters,
            and the second array represents the corresponding labels.

        Notes:
        -------
        This method follows these steps:
        1. If the 'save' attribute is True, it removes the content of the processed images directory.
        2. It loads images from the specified 'origen_path' directory.
        3. It processes the images and extracts character images.
        4. If the 'save' attribute is True, it saves the processed images to the specified 'new_path' directory.
        5. It generates numpy arrays from the processed character images and returns them.
        """
        
        # Remove the content of the processed images directory:
        if self.save:
            ff.remove_directory_content(IMAGES_PATH+self.new_path)
        
        # Load images:
        dirs: List[str] = self.__get_all_directories(self.origen_path)
        images_dir: List[tuple[List[MatLike],str]] = self.__charge_train_images(dirs)
        
        # Process images and obtain characters:
        imgs_preps: List[ImagePreprocessor] = self.__create_images_preprocessors(images_dir)
        characters: List[tuple[List[MatLike],str]] = [
            (prep.proccess_images(),prep.path) 
            for prep in imgs_preps
        ]
        
        # Save processed images:
        if self.save:
            self.__save_characters(characters,self.origen_path,self.new_path)
        
        # Create arrays:
        return self.__generate_arrays(characters)