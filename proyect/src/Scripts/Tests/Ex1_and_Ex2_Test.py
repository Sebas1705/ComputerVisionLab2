from Scripts.Executors import Ex1,Ex2
from Classes.Ex1AndEx2.CharactersLoader import CharactersLoader
from Settings import *
import threading
import pickle
import sys

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
    

def test(ex1: bool|None=None,classifier_name:str="All") -> None:
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
    
    if DEBUG_MODE:
        print(f"\t\t\t\tC_train: {sys.getsizeof(c_train):.2f} Bytes")
        print(f"\t\t\t\tE_train: {sys.getsizeof(e_train):.2f} Bytes")
        print(f"\t\t\t\tC_test: {sys.getsizeof(c_test):.2f} Bytes")
        print(f"\t\t\t\tE_test: {sys.getsizeof(e_test):.2f} Bytes")
        
    # Verify arrays ar not None
    if c_train is None or e_train is None or c_test is None or e_test is None:
        raise ValueError(f"{threading.current_thread().name}->Failed to load train or test arrays.")
    
    #Execute exercise 1 and 2:
    if ex1 is None:
        Ex1.exec1(c_train,e_train,c_test,e_test,classifier_name)
        Ex2.exec2(c_train,e_train,c_test,e_test,classifier_name)
    elif ex1:
        Ex1.exec1(c_train,e_train,c_test,e_test,classifier_name)
    else:
        Ex2.exec2(c_train,e_train,c_test,e_test,classifier_name)
    