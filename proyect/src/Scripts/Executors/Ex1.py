import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Classes.Ex1AndEx2.ClassifierTester import ClassifierTester
import sys
from Settings import DEBUG_MODE

def exec1(
    c_train: np.ndarray,
    e_train: np.ndarray,
    c_test: np.ndarray,
    e_test: np.ndarray,    
    classifier_name: str = 'All'    
) -> str:
    #Create LDA and CRs:
    lda = LinearDiscriminantAnalysis()
    lda.fit(c_train,e_train)
    CR: np.ndarray = lda.transform(c_train)   
    CR_TEST: np.ndarray = lda.transform(c_test)
    
    if DEBUG_MODE:
        print(f"\t\t\t\tCR(LDA): {sys.getsizeof(CR):.2f} Bytes")
        print(f"\t\t\t\tCR_TEST(LDA): {sys.getsizeof(CR_TEST):.2f} Bytes")
    
    #Test classifiers:
    tester = ClassifierTester('LDA',CR,e_train,CR_TEST,e_test,classifier_name)
    return tester.test_classifier()