import threading

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from Classes.Ex3.PanelsLoader import PanelsLoader
from Scripts.Executors import Ex1,Ex2
from Classes.Ex1AndEx2.CharactersLoader import CharactersLoader
from Common.Settings import DEBUG_MODE

#Globals arrays:
c_train = e_train = c_test = e_test = None

def get_trains(save_chars: bool) -> None:
    #Train arrays:
    global c_train, e_train
    imgs_train_loader = CharactersLoader("train_ocr_origen\\","train_ocr_new\\",save_chars)
    c_train,e_train = imgs_train_loader.get_arrays()
    if DEBUG_MODE:
        print(f"{threading.current_thread().name}->Trains characters loaded")

def get_tests(save_chars: bool) -> None:
    #Test arrays:
    global c_test,e_test
    imgs_test_loader = CharactersLoader("validation_ocr_origen\\","validation_ocr_new\\",save_chars)
    c_test,e_test = imgs_test_loader.get_arrays()
    if DEBUG_MODE:
        print(f"{threading.current_thread().name}->Tests characters loaded")
    

def test() -> None:
    
    global c_train, e_train, c_test, e_test
    
    #Check if characters will be saved or not
    save_characters = False
    
    #Get images and generate train and test arrays:
    arrys: list[threading.Thread] = [
        threading.Thread(target=get_trains, args=(save_characters,), name="Get trains"),
        threading.Thread(target=get_tests, args=(save_characters,), name="Get tests")
    ]
    [arr.start() for arr in arrys]
    [arr.join() for arr in arrys]
    
    #Create PCA and CRs:
    pca = PCA(n_components=100)
    pca.fit(c_train)
    CR: np.ndarray = pca.transform(c_train)
    CR_TEST: np.ndarray = pca.transform(c_test)
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(CR,e_train)
    
    #Charge all test images:
    p = PanelsLoader(knn,pca,"test_ocr_panels_origen\\","test_ocr_panels_new\\",True)
    images = p.get_proccess_panels()
