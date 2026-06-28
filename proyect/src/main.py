from Scripts.Tests.Ex1_and_Ex2_Test import test as exs_1_2_test
from Scripts.Tests.Ex3_Test import test as ex_3_test
from Scripts.Tests.Ex4_Test import test as ex_4_test
#Main function:
def main() -> None:
    exs_1_2_test(ex1=None,classifier_name="All")
    ex_3_test()
    ex_4_test()
    
if __name__ == '__main__':
    main()