from Classes.Ex1AndEx2.CharactersLoader import CharactersLoader
from Scripts.Executors import Ex3


def test() -> None:
    
    #Check if characters will be saved or not:
    save_characters = False
    
    #Load and get c and e arrays:
    imgs_train_loader = CharactersLoader("train_ocr_origen\\","train_ocr_new\\",save_characters)
    c_train,e_train = imgs_train_loader.get_arrays()
    
    #Executate the third exercise:
    Ex3.exec3(c_train,e_train)
    
