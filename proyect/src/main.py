from Scripts.Tests.Ex1_and_Ex2_Test import test as exs_1_2_test
from Scripts.Tests.Ex3_Test import test as ex_3_test

#Main function of proyect:
def main() -> None:
    #exs_1_2_test(ex1=False,classifier_name="GaussianNB")
    ex_3_test()
    
    
if __name__ == '__main__':
    main()