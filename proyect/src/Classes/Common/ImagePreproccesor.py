from typing import List, Sequence
from cv2.typing import MatLike
from abc import ABC,abstractmethod
import cv2

class ImagePreproccesor(ABC):
    
    def __init__(
        self,
        path: str,
        images: List[MatLike] = []
    ) -> None:
        """
        Initialize ImagePreprocessor object.

        Parameters:
        -----------
        path : str 
            The path to the directory containing the images.    
        images : List[MatLike], optional 
            A list of images to be processed. Defaults to an empty list.

        Returns:
        --------
        None
        """
    
        self.images: List[MatLike] = images
        self.n_images: int = len(images)
        self.path: str = path
        
    def convert_grayscale(
        self
    ) -> None:
        """
        Converts the list of images from color to grayscale.

        Parameters:
        -----------
        self : ImagePreprocessor
            The instance of the ImagePreprocessor class.

        Returns:
        --------
        None
        """
        
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
        
    def adaptative_umbralize(
        self
    ) -> List[MatLike]:
        """
        Applies an adaptive threshold to each grayscale image in the list.
        The threshold value is calculated based on the mean and standard deviation of the neighborhood area.
        The method is more suitable for images with varying illumination conditions.

        Parameters:
        -----------
        self : ImagePreprocessor
            The instance of the ImagePreprocessor class.

        Returns:
        --------
        List[MatLike]
            A list of thresholded images. Each image is a binary image where pixels with intensity greater than the threshold are set to 255, and the others are set to 0.
        """
        
        return [
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)
            for gray in self.images
        ]
        
    def binary_umbralize(
        self
    ) -> List[MatLike]:
        """
        Applies an adaptive threshold to each grayscale image in the list.
        The threshold value is calculated based on the mean and standard deviation of the neighborhood area.
        The method is more suitable for images with varying illumination conditions.

        Parameters:
        -----------
        self : ImagePreprocessor
            The instance of the ImagePreprocessor class.

        Returns:
        --------
        List[MatLike]
            A list of thresholded images. Each image is a binary image where pixels with intensity greater than the threshold are set to 255, and the others are set to 0.
        """
        
        return [
            cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
            for gray in self.images
        ]
        
    def get_contours(
        self,
        threshs: List[MatLike]
    ) -> List[tuple[Sequence[MatLike],MatLike]]:
        """
        Finds contours in a list of thresholded images.

        Parameters:
        -----------
        self : ImagePreprocessor
            The instance of the ImagePreprocessor class.
        threshs : List[MatLike]
            A list of thresholded images. Each image is a binary image where pixels with intensity greater than the threshold are set to 255, and the others are set to 0.

        Returns:
        --------
        List[tuple[Sequence[MatLike],MatLike]]
            A list of tuples. Each tuple contains a list of contours found in the corresponding thresholded image and the thresholded image itself.
        """
        
        return [
            (cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],thresh)
            for thresh in threshs
        ]
        
    @abstractmethod
    def proccess_images(
        self
    ) -> List[MatLike]:
        pass