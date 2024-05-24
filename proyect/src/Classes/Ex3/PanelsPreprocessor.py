from typing import List, Sequence
from cv2.typing import MatLike,Rect
from Classes.Common.ImagePreproccesor import ImagePreproccesor
from sklearn.linear_model import RANSACRegressor
import numpy as np
import cv2

class PanelsPreproccesor(ImagePreproccesor):
    
    def detect_characters(self,image):
        gray = image
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if 0.2 < aspect_ratio < 1.0:
                characters.append((x + w // 2, y + h // 2))  # Agregar centroide del rectángulo como caracter

        return np.array(characters)

    def detect_lines_with_ransac(self,characters):
        line_model = RANSACRegressor(min_samples=2)

        lines = []
        while characters.shape[0] > 1:
            line_model.fit(characters[:, 0].reshape(-1, 1), characters[:, 1])
            inliers = line_model.inlier_mask_

            lines.append(line_model.estimator_.coef_)
            characters = characters[~inliers]

        return lines

    def classify_characters(self,image, characters):
        classified_characters = []
        for character in characters:
            # Suponiendo que tienes una función para clasificar caracteres
            # classified_character = classifier.predict(character)
            classified_character = 'A'  # Suponiendo que 'A' representa un ejemplo de clasificación
            classified_characters.append(classified_character)

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

        
        characters = [self.detect_characters(thresh) for thresh in threshs]
        
        lines = [self.detect_lines_with_ransac(chars) for chars in characters]
        
        classified_characters = [self.classify_characters(self.images[i],characters[i]) for i in range(len(characters))]
        
        strings = [
            f"Nº{i}"+self.generate_output_string(classified_characters[i], self.images[i].shape[1])
            for i in range(len(classified_characters))
        ]
        
        print("\n------\n".join(strings))
        
        return []