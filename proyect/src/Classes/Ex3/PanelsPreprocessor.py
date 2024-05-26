import string
from typing import List 
from cv2.typing import MatLike 
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from Classes.Common.ImagePreproccesor import ImagePreproccesor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
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
    
    def classify_characters(self,characters:List[MatLike]):
        classified_characters = []
        for character in characters:
            image = self.pca.transform(np.ravel(np.array(character)).reshape(1,-1))
            classified_character = self.classifier.predict(image)
            classified_characters.append(self.letters[classified_character[0]])
        print(classified_characters)
        return classified_characters


    def proccess_images(
        self
    ) -> List[MatLike]:
        
        #Umbralizar:
        self.convert_grayscale()
        clahe: cv2.CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.images = [clahe.apply(img) for img in self.images]
        
        images: List[MatLike]  = [
            cv2.threshold(img,np.mean(img), 255, cv2.THRESH_BINARY)[1]
            for img in self.images
        ]
        
        mser: cv2.MSER = cv2.MSER_create()
        classified_characters = []
        for i in range(len(images)):
        
            # Detectar regiones MSER
            regiones, _ = mser.detectRegions(images[i])

            # Crear una copia de la imagen original para dibujar las regiones detectadas
            imagen_con_regiones = cv2.cvtColor(self.images[i].copy(),cv2.COLOR_GRAY2RGB)

            # Dibujar las regiones detectadas
            centros = []
            characters = []
            for region in regiones:
                x,y,w,h=cv2.boundingRect(region)
                if 0.2 < w/h < 1.2 and 150 < w*h < 1000:
                    M = cv2.moments(region)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centros.append((cX, cY))
                    cv2.rectangle(imagen_con_regiones,(x,y),(x+w, y+h),(0,255,0),2)
                    characters.append(np.ravel(np.array(cv2.resize(self.images[i],CHARS_SIZE))).reshape(1,-1))
            images[i] = imagen_con_regiones
            centros = np.array(centros)
            
            lineas = []
            while len(centros) > 4:  # Al menos 5 puntos para encontrar una línea
                # Ajustar una línea usando RANSAC
                X = centros[:, 0].reshape(-1, 1)
                y = centros[:, 1]
                ransac = make_pipeline(PolynomialFeatures(1), RANSACRegressor(LinearRegression(), residual_threshold=10))
                ransac.fit(X, y)
                
                # Obtener los inliers
                inliers = ransac.named_steps['ransacregressor'].inlier_mask_
                linea_actual = centros[inliers]
                
                # Guardar la línea encontrada
                lineas.append(linea_actual)
                
                # Eliminar los inliers del conjunto de centros
                centros = centros[~inliers]
            
            imagen_con_lineas = images[i].copy()
            for linea in lineas:
                [vx, vy, x, y] = cv2.fitLine(linea, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((imagen_con_lineas.shape[1] - x) * vy / vx) + y)
                cv2.line(imagen_con_lineas, (imagen_con_lineas.shape[1] - 1, righty), (0, lefty), (0, 0, 255), 1)
            images[i] = imagen_con_lineas
            
            classified_characters.append(characters)
            
        clc = []
        for characters in classified_characters:
            clc.append(self.classify_characters(characters))
         
            
        # strings = [
        #     f"Nº{i}"+self.generate_output_string(classified_characters[i], self.images[i].shape[1])
        #     for i in range(len(classified_characters))
        # ]
        
        # print("\n------\n".join(strings))
        
        return images