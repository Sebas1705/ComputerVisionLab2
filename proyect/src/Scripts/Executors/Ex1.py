import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from Classes.Ex1AndEx2.ClassifierTester import ClassifierTester

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
    #Test classifiers:
    tester = ClassifierTester('LDA',CR,e_train,CR_TEST,e_test,classifier_name)
    return tester.test_classifier()