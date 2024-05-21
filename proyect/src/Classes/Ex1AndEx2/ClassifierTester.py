from threading import Thread
from typing import Any, List
from Settings import FILES_PATH
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import Common.FileFuncs as ff
import numpy as np
import time

class ClassifierTester:
    
    def __init__(
        self,
        rd_name: str,
        cr_train: np.ndarray,
        e_train: np.ndarray,
        cr_test: np.ndarray,
        e_test: np.ndarray,
        clf_name: str = "All",
    ) -> None:
        """
        Initialize a ClassifierTester instance.

        Parameters:
        -----------
        rd_name : str 
            The name of the dataset.
        cr_train : np.ndarray 
            The training dataset for the features.
        e_train : np.ndarray 
            The training dataset for the labels.
        cr_test : np.ndarray 
            The testing dataset for the features.
        e_test : np.ndarray 
            The testing dataset for the labels.
        clf_name : str, optional
            The name of the classifier to be used. If 'All', all classifiers will be tested. 
            The default is 'All'.

        Returns:
        --------
            None

        Raises:
        -------
            None
        """
    
        self.rd_name: str = rd_name
        self.cr_train: np.ndarray = cr_train
        self.e_train: np.ndarray = e_train
        self.cr_test: np.ndarray = cr_test
        self.e_test: np.ndarray = e_test
        self.clf_name: str = clf_name
        self.CLASSIFIERS: dict[str,Any] = {
            'SVC': SVC(),
            'RandomForest': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'LogisticRegression': LogisticRegression(max_iter=1000),
            'DecisionTree': DecisionTreeClassifier(),
            'GaussianNB': GaussianNB()
        }
        self.mets: List[str] = []

    def train_and_metrics(
        self,
        clf_name:str
    ) -> str:
        """
        This method trains a classifier, makes predictions, and calculates metrics.

        Parameters:
        -----------
        clf_name : str 
            The name of the classifier to be used. It should be a key in the CLASSIFIERS dictionary.

        Returns:
        --------
        classification report: str
            A string containing the classification report and the time taken for the operation.

        Raises:
        -------
            KeyError: If the clf_name is not a key in the CLASSIFIERS dictionary.
        """
        
        start = time.time()
        #Create and train classifier:
        clf = self.CLASSIFIERS[clf_name]
        clf.fit(self.cr_train,self.e_train)
        #Predict:
        pred=clf.predict(self.cr_test)
        #Metrics:
        mets: str = metrics.classification_report(self.e_test,pred)
        ff.create_txt(FILES_PATH+f"{self.rd_name}_Metrics/"+clf_name+".txt",mets)
        self.mets.append(mets + "\n------------------\n" + f"\n\tTime: {(time.time()-start):.2f}")
        print(f"Finish {self.rd_name}-{clf_name}")

    def test_classifier(
        self
    ) -> str:
        """
        This method tests a classifier by training and evaluating it using different algorithms.
        If 'All' is selected, it will train and evaluate all classifiers in parallel.
        If a specific classifier is selected, it will only train and evaluate that one.

        Parameters:
        -----------
        self : ClassifierTester 
            The instance of the ClassifierTester class.

        Returns:
        --------
        str: str
            A string containing the classification reports and the time taken for 
            each operation. If 'All' classifiers are tested, the reports are separated 
            by a newline character.

        Raises:
        -------
            None
        """
        
        if self.clf_name == 'All':
            mets: List[Thread] = [
                Thread(target=self.train_and_metrics,args=('SVC',)),
                Thread(target=self.train_and_metrics,args=('RandomForest',)),
                Thread(target=self.train_and_metrics,args=('KNN',)),
                Thread(target=self.train_and_metrics,args=('LogisticRegression',)),
                Thread(target=self.train_and_metrics,args=('DecisionTree',)),
                Thread(target=self.train_and_metrics,args=('GaussianNB',))
            ]
            [met.start() for met in mets]
            [met.join() for met in mets]
            return "\n".join(self.mets)
        else:
            return self.train_and_metrics(self.clf_name)
