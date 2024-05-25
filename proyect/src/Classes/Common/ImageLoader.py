from typing import List
from cv2.typing import MatLike
from abc import ABC,abstractmethod
from Classes.Common.ImagePreproccesor import ImagePreproccesor
from Common.Settings import *
import Common.FileFuncs as ff


class ImageLoader(ABC):
    
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
        
    @abstractmethod
    def get_all_directories(
        self,
        dir:str
    ) -> List[str]:
        pass
    
    def charge_images(
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
    
    @abstractmethod
    def save_images(
        self,
        images:List[tuple[List[MatLike],str]],
        image_dir: str,
        new_path: str
    ) -> None:
        pass
    
    @abstractmethod
    def create_preprocessors(
        self,
        images_dir: List[tuple[List[MatLike],str]]
    ) -> List[ImagePreproccesor]:
        pass