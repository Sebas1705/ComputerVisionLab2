from Scripts.Executors import Ex1
from sklearn.decomposition import PCA
import numpy as np

def exec2(
    c_train: np.ndarray,
    e_train: np.ndarray,
    c_test: np.ndarray,
    e_test: np.ndarray,   
    clasifier_name: str = 'All'
) -> str:
    """
    This function is the main entry point for the OCR (Optical Character Recognition) 
    system. It prepares the training and testing datasets, applies Linear Discriminant 
    Analysis (LDA) for dimensionality reduction, and then tests the specified classifier.

    Parameters:
    -----------
    clasifier_name : str, optional 
        The name of the classifier to be tested. It should be one of the following:
        'SVC', 'RandomForest', 'KNN', 'LogisticRegression', 'DecisionTree', 'GaussianNB' or 'All' to test all.
        Defaults to 'All'.
        
    Returns:
    -----------
    str
        Classification report for the specified classifier
    """
    
    
    
    #Create LDA and CR:
    pca = PCA(n_components=200)
    pca.fit(c_train)
    CR: np.ndarray = pca.transform(c_train)
    
    #Test arrays:
    
    CR_TEST: np.ndarray = pca.transform(c_test)
   
    #Test clasifiers:
    if clasifier_name == "All":
        mets = ""
        mets += Ex1.__test_clasifier('SVC','PCA',CR,e_train,CR_TEST,e_test)
        mets += Ex1.__test_clasifier('RandomForest','PCA',CR,e_train,CR_TEST,e_test)
        mets += Ex1.__test_clasifier('KNN','PCA',CR,e_train,CR_TEST,e_test)
        mets += Ex1.__test_clasifier('LogisticRegression','PCA',CR,e_train,CR_TEST,e_test)
        mets += Ex1.__test_clasifier('DecisionTree','PCA',CR,e_train,CR_TEST,e_test)
        mets += Ex1.__test_clasifier('GaussianNB','PCA',CR,e_train,CR_TEST,e_test)
        return mets
    return Ex1.__test_clasifier(clasifier_name,'PCA',CR,e_train,CR_TEST,e_test)