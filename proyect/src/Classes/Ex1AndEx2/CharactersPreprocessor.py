from typing import List, Sequence
from cv2.typing import MatLike
from Classes.Common.ImagePreproccesor import ImagePreproccesor
from Common.Settings import CHARS_SIZE
import cv2

class CharactersPreprocessor(ImagePreproccesor):
    
    def proccess_images(
        self
    ) -> List[MatLike]:
        
        self.convert_grayscale()
        threshs: List[MatLike] = self.adaptative_umbralize()
        countours: List[Sequence[MatLike],] = self.get_contours(threshs)
        
        characters: List[MatLike] = []
        for countours_s,img in countours:
            for contour in countours_s:
                x, y, w, h = cv2.boundingRect(contour)
            characters.append(cv2.resize(img[y:y+h+2,x:x+w+2],CHARS_SIZE))
                
        return characters