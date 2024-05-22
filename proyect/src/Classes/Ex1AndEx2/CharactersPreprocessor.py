from typing import List, Sequence
from cv2.typing import MatLike
from Common import FileFuncs as ff
import cv2

class CharactersPreprocessor:
    
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
        
    def __convert_grayscale(
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
    
    def __adaptative_umbralize(
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
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            for gray in self.images
        ]
        
    def __get_contours(
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
            
    def proccess_images(
        self
    ) -> List[MatLike]:
        """
        Processes the images in the ImagePreprocessor instance.

        Converts the images to grayscale, applies an adaptive threshold, finds contours,
        and resizes the detected characters to a standard size.

        Parameters:
        -----------
        self : ImagePreprocessor
            The instance of the ImagePreprocessor class.

        Returns:
        --------
        List[MatLike]
            A list of processed character images. Each image is a 25x25 grayscale image.
        """
        
        self.__convert_grayscale()
        threshs: List[MatLike] = self.__adaptative_umbralize()
        countours: List[Sequence[MatLike],] = self.__get_contours(threshs)
        
        characters: List[MatLike] = []
        for countours_s,img in countours:
            for contour in countours_s:
                x, y, w, h = cv2.boundingRect(contour)
            characters.append(cv2.resize(img[y:y+h+2,x:x+w+2],(25,25)))
                
        return characters