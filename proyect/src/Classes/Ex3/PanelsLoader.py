from numpy import ndarray
from typing import List, override
from cv2.typing import MatLike
from sklearn.decomposition import PCA
from Common.Settings import IMAGES_PATH
from Classes.Ex1AndEx2.CharactersPreprocessor import CharactersPreprocessor
from sklearn.neighbors import KNeighborsClassifier
from Classes.Common.ImageLoader import ImageLoader
from Classes.Common.ImagePreproccesor import ImagePreproccesor
import os
import Common.FileFuncs as ff
import numpy as np
from Classes.Ex3.PanelsPreprocessor import PanelsPreproccesor

class PanelsLoader(ImageLoader):
    
    def __init__(
        self,
        classifier: KNeighborsClassifier,
        pca: PCA,
        origen_path: str,
        new_path: str,
        save: bool = False
    ) -> None:
        ImageLoader.__init__(self,origen_path,new_path,save)
        self.classifier: KNeighborsClassifier = classifier
        self.pca: PCA = pca
    
    @override
    def get_all_directories(
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
        
        return [dir]
        
    @override
    def save_images(
        self,
        images: List[tuple[List[MatLike],str]],
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
        
        for imgs,str in images:
            path = IMAGES_PATH+new_path+str
            path = path.split(image_dir)
            path = path[0]+path[1]
            print(path,"-",len(imgs))
            ff.save_images(imgs,path)
       
    @override   
    def create_preprocessors(
        self,
        images_dir: List[tuple[List[MatLike],str]]
    ) -> List[ImagePreproccesor]:
        """
        This function creates a list of CharactersPreprocessor objects. Each CharactersPreprocessor object is initialized with a tuple containing
        a list of images and their corresponding directory path.

        Parameters:
        -----------
        images_dir : List[tuple[List[MatLike],str]]
            A list of tuples, where each tuple contains a list of images and their corresponding directory path.
            The first element of the tuple is a list of images, and the second element is a string representing the directory path.

        Returns:
        -----------
        List[CharactersPreprocessor]
            A list of CharactersPreprocessor objects, where each object is initialized with a tuple from the input list.
            The CharactersPreprocessor objects are responsible for processing the images and extracting character images.

        Notes:
        -------
        This function iterates over the input list of tuples, creates an CharactersPreprocessor object for each tuple,
        and appends the object to a list. Finally, it returns the list of CharactersPreprocessor objects.

        The CharactersPreprocessor class is assumed to have a constructor that takes two parameters:
        - directory path (str)
        - list of images (List[MatLike])

        The CharactersPreprocessor class is responsible for processing the images and extracting character images.
        """
        
        imgs_preps: List[PanelsPreproccesor] = []
        for tp in images_dir:
            img_prep: PanelsPreproccesor = PanelsPreproccesor(self.classifier,self.pca,tp[1],tp[0])
            imgs_preps.append(img_prep)
        return imgs_preps
  
    def get_proccess_panels(
        self,
    ) -> List[MatLike] :
        
        # Remove the content of the processed images directory:
        if self.save:
            ff.remove_directory_content(IMAGES_PATH+self.new_path)
        
        # Load images:
        dirs: List[str] = self.get_all_directories(self.origen_path)
        images_dir: List[tuple[List[MatLike],str]] = self.charge_images(dirs)
        
        # Process images and obtain characters:
        imgs_preps: List[PanelsPreproccesor] = self.create_preprocessors(images_dir)
        proccess_images: List[tuple[List[MatLike],str]] = [
            (prep.proccess_images(),prep.path) 
            for prep in imgs_preps
        ]
        
        # Save processed images:
        if self.save:
            self.save_images(proccess_images,self.origen_path,self.new_path)
        
        # return images:
        imgs: List[MatLike] = []
        [imgs.append(i) for img,_ in proccess_images for i in img]
        return imgs