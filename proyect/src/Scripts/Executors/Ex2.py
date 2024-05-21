import numpy as np
from sklearn.decomposition import PCA
from Classes.Ex1AndEx2.ClassifierTester import ClassifierTester

def exec2(
    c_train: np.ndarray,
    e_train: np.ndarray,
    c_test: np.ndarray,
    e_test: np.ndarray,   
    classifier_name: str = 'All'
) -> str:
    #Create PCA and CRs:
    pca = PCA(n_components=200)
    pca.fit(c_train)
    CR: np.ndarray = pca.transform(c_train)
    CR_TEST: np.ndarray = pca.transform(c_test)
    #Test classifiers:
    tester = ClassifierTester('PCA',CR,e_train,CR_TEST,e_test,classifier_name)
    return tester.test_classifier()