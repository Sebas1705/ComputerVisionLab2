import sys
import numpy as np
from sklearn.decomposition import PCA
from Classes.Ex1AndEx2.ClassifierTester import ClassifierTester
from Settings import DEBUG_MODE

def exec2(
    c_train: np.ndarray,
    e_train: np.ndarray,
    c_test: np.ndarray,
    e_test: np.ndarray,   
    classifier_name: str = 'All'
) -> str:
    #Create PCA and CRs:
    pca = PCA(n_components=100)
    pca.fit(c_train)
    CR: np.ndarray = pca.transform(c_train)
    CR_TEST: np.ndarray = pca.transform(c_test)
    
    if DEBUG_MODE:
        print(f"\t\t\t\tCR(PCA): {sys.getsizeof(CR):.2f} Bytes")
        print(f"\t\t\t\tCR_TEST(PCA): {sys.getsizeof(CR_TEST):.2f} Bytes")
    
    #Test classifiers:
    tester = ClassifierTester('PCA',CR,e_train,CR_TEST,e_test,classifier_name)
    return tester.test_classifier()