from Scripts.Executors import Ex1,Ex2
from Classes.Ex1AndEx2.ImageLoader import ImageLoader
import threading

#Globals arrays:
c_train = e_train = c_test = e_test = None

def get_trains(save_chars: bool) -> None:
    #Train arrays:
    global c_train, e_train
    imgs_train_loader = ImageLoader("train_ocr_origen\\","train_ocr_new\\",save_chars)
    c_train,e_train = imgs_train_loader.get_arrays()

def get_tests(save_chars: bool) -> None:
    #Test arrays:
    global c_test,e_test
    imgs_test_loader = ImageLoader("validation_ocr_origen\\","validation_ocr_new\\",save_chars)
    c_test,e_test = imgs_test_loader.get_arrays()
    

def main() -> None:
    global c_train, e_train, c_test, e_test
    
    #Check if characters will be saved or not
    save_characters = False
    
    #Get images and generate train and test arrays:
    arrys: list[threading.Thread] = [
        threading.Thread(target=get_trains, args=(save_characters,)),
        threading.Thread(target=get_tests, args=(save_characters,))
    ]
    [arr.start() for arr in arrys]
    [arr.join() for arr in arrys]
    
    # Verify arrays ar not None
    if c_train is None or e_train is None or c_test is None or e_test is None:
        raise ValueError("Failed to load train or test arrays.")
    
    #Execute exercise 1 and 2:
    classifier_name = "All"
    execs: list[threading.Thread] = [
        threading.Thread(target=Ex1.exec1,args=(c_train.copy(),e_train.copy(),c_test.copy(),e_test.copy(),classifier_name)),
        threading.Thread(target=Ex2.exec2,args=(c_train.copy(),e_train.copy(),c_test.copy(),e_test.copy(),classifier_name))
    ]
    [exec.start() for exec in execs]
    [exec.join() for exec in execs]
    

if __name__ == "__main__":
    main()