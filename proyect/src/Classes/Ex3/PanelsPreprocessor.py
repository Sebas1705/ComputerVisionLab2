import string
from typing import List, Sequence
from cv2.typing import MatLike,Rect
from sklearn.decomposition import PCA
from Classes.Common.ImagePreproccesor import ImagePreproccesor
from sklearn.linear_model import RANSACRegressor
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import cv2
from Common.Settings import CHARS_SIZE

class PanelsPreproccesor(ImagePreproccesor):
    
    def __init__(
        self,
        classifier: KNeighborsClassifier,
        pca: PCA,
        path: str,
        images: List[MatLike] = [],
    ) -> None:
        ImagePreproccesor.__init__(self,path,images)
        self.classifier: KNeighborsClassifier = classifier
        self.pca = pca
        self.letters = string.digits + string.ascii_uppercase + string.ascii_lowercase
    
    def detect_characters(self,image) -> tuple[List[tuple[float,float]],List[MatLike]]:
        gray = image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, img = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        characters = []
        images = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 1.0:
                characters.append((x + w // 2, y + h // 2))  # Agregar centroide del rectángulo como caracter
                images.append(cv2.resize(image[y:y+h+2,x:x+w+2],CHARS_SIZE))
                
        return np.array(characters),images

    def detect_lines_with_ransac(self,characters):
        line_model = RANSACRegressor(min_samples=2)

        lines = []
        while characters.shape[0] > 1:
            line_model.fit(characters[:, 0].reshape(-1, 1), characters[:, 1])
            inliers = line_model.inlier_mask_

            lines.append(line_model.estimator_.coef_)
            characters = characters[~inliers]

        return lines

    def classify_characters(self,characters:List[MatLike]):
        classified_characters = []
        for character in characters:
            image = self.pca.transform(np.ravel(np.array(character)).reshape(1,-1))
            classified_character = self.classifier.predict(image)
            classified_characters.append(self.letters[classified_character[0]])
        print(classified_characters)
        return classified_characters

    def generate_output_string(self,classified_characters, image_width):
        output_string = ''
        for idx, char in enumerate(classified_characters):
            if (idx + 1) % image_width == 0:
                output_string += '+'
            else:
                output_string += char

        return output_string

    def proccess_images(
        self
    ) -> List[MatLike]:
        
        #Umbralizar:
        self.convert_grayscale()
        threshs: List[MatLike] = self.adaptative_umbralize()
        # Invertir la imagen para tener las letras en blanco y el fondo en negro

        for thresh in threshs:
            characters,imgs = self.detect_characters(thresh)
            lines = self.detect_lines_with_ransac(characters)
            classified_characters = self.classify_characters(imgs)
        
        # strings = [
        #     f"Nº{i}"+self.generate_output_string(classified_characters[i], self.images[i].shape[1])
        #     for i in range(len(classified_characters))
        # ]
        
        # print("\n------\n".join(strings))
        
        return []