from typing import List, Sequence
from cv2.typing import MatLike
from Common import FileFuncs as ff
import cv2

class ImagePreprocessor:
    
    def __init__(
        self,
        path: str,
        images: List[MatLike] = []
    ) -> None:
        self.images: List[MatLike] = images
        self.n_images: int = len(images)
        self.path: str = path
        
    def __convert_grayscale(
        self
    ) -> None:
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images]
    
    def __adaptative_umbralize(
        self
    ) -> List[MatLike]:
        return [
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            for gray in self.images
        ]
        
    def __get_contours(
        self,
        threshs: List[MatLike]
    ) -> List[Sequence[MatLike]]:
        return [
            cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            for thresh in threshs
        ]
            
    def proccess_images(
        self
    ) -> List[MatLike]:
        
        self.__convert_grayscale()
        threshs: List[MatLike] = self.__adaptative_umbralize()
        countours: List[Sequence[MatLike]] = self.__get_contours(threshs)
        
        characters: List[MatLike] = []
        for countours_s in countours:
            for contour in countours_s:
                x, y, w, h = cv2.boundingRect(contour)
                characters.append(cv2.resize(self.images[0][y:y+h,x:x+w],(25,25)))
        
        return characters