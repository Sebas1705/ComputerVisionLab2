from Scripts.Executors import Ex1
from Scripts.Executors import Ex2

if __name__ == "__main__":
    #Train arrays:
    c_train,e_train = Ex1.__get_arrays("train_ocr_origen\\","train_ocr_new\\")
    #Test arrays:
    c_test,e_test = Ex1.__get_arrays("validation_ocr_origen\\","validation_ocr_new\\")
    Ex1.exec1(c_train,e_train,c_test,e_test)
    Ex2.exec2(c_train,e_train,c_test,e_test)


