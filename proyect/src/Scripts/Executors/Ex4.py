from typing import List
from cv2.typing import MatLike
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from Classes.Ex3.PanelsLoader import PanelsLoader
from p1.src.main import p1

def exec4(
    c_train,
    e_train,
) -> List[MatLike]:
    p1()
    #Create PCA and CRs:
    pca = PCA(n_components=100)
    pca.fit(c_train)
    CR: np.ndarray = pca.transform(c_train)
    #CR_TEST: np.ndarray = pca.transform(c_test)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(CR,e_train)
    
    #Charge all test images:
    p = PanelsLoader(knn,pca,"p1_images\\i_final_cropped","panels_p1_new\\",True)
    return p.get_proccess_panels()