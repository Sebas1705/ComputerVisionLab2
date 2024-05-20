import numpy as np
from Classes.Ex1.ImagePreprocessor import ImagePreprocessor
from Settings import IMAGES_PATH,SAVE_PROCESED_IMAGES,FILES_PATH
import Common.FileFuncs as ff
from typing import List
from cv2.typing import MatLike
import os
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def __get_all_directories(dir:str) -> List[str]:
    """
    This function retrieves all directories within a given directory path.

    Parameters:
    -----------
    dir : str
        The directory path from which to retrieve subdirectories.

    Returns:
    -----------
    List[str]
        A list of all subdirectory paths within the given directory.

    Notes:
    -------
    This function first retrieves all subdirectories within the given directory.
    Then, it appends two extra subdirectories (dirs[-2] and dirs[-1]) to the list.
    Finally, it iterates over these extra subdirectories and retrieves all subdirectories within them.
    The function returns a list of all subdirectory paths.
    """
    
    dirs: List[str] = [dir+d for d in os.listdir(IMAGES_PATH+dir)]
    extras: List[str] = [dirs[-2],dirs[-1]]
    dirs = dirs[:-2]
    for ext in extras:
        for dir in os.listdir(IMAGES_PATH+ext):
            dirs.append(ext+"\\"+dir)
    return dirs

def __charge_train_images(dirs:List[str]) -> List[tuple[List[MatLike],str]]:
    """
    This function loads images from the given directories and returns a list of tuples.
    Each tuple contains a list of images and their corresponding directory path.

    Parameters:
    -----------
    dirs : List[str]
        A list of directory paths where the images are located.

    Returns:
    -----------
    List[tuple[List[MatLike],str]]
        A list of tuples. Each tuple contains a list of images and their corresponding directory path.
        The images are loaded using the `read_images` function from the `ff` module.

    Notes:
    -------
    This function iterates over the input list of directory paths, loads the images using the `read_images` function,
    and appends a tuple containing the loaded images and their corresponding directory path to the output list.
    Finally, it returns the output list.
    """
    
    images_dir: List[tuple[List[MatLike],str]] = []
    for dir in dirs:
        images_dir.append((ff.read_images(IMAGES_PATH+dir),dir))
    return images_dir

def __create_images_preprocessors(
    images_dir: List[tuple[List[MatLike],str]]
) -> List[ImagePreprocessor]:
    """
    This function creates a list of ImagePreprocessor objects. Each ImagePreprocessor object is initialized with a tuple containing
    a list of images and their corresponding directory path.

    Parameters:
    -----------
    images_dir : List[tuple[List[MatLike],str]]
        A list of tuples, where each tuple contains a list of images and their corresponding directory path.

    Returns:
    -----------
    List[ImagePreprocessor]
        A list of ImagePreprocessor objects, where each object is initialized with a tuple from the input list.

    Notes:
    -------
    This function iterates over the input list of tuples, creates an ImagePreprocessor object for each tuple,
    and appends the object to a list. Finally, it returns the list of ImagePreprocessor objects.
    """

    imgs_preps: List[ImagePreprocessor] = []
    for tp in images_dir:
        img_prep: ImagePreprocessor = ImagePreprocessor(tp[1],tp[0])
        imgs_preps.append(img_prep)
    return imgs_preps

def __save_characters(
    characters: List[tuple[List[MatLike],str]],
    image_dir: str,
    new_path: str
) -> None:
    """
    This function saves the processed character images into separate directories for uppercase and lowercase characters.

    Parameters:
    -----------
    characters : List[tuple[List[MatLike],str]]
        A list of tuples, where each tuple contains a list of character images and their corresponding directory path.
    image_dir : str
        The directory path where the original images are located.
    new_path : str
        The directory path where the processed images will be saved.

    Returns:
    -----------
    None
        This function does not return any value. It saves the processed character images into separate directories.

    Notes:
    -------
    The function creates two directories within the 'new_path' directory: 'mayus' and 'minus'.
    It then iterates over the list of character images, extracts the directory path from the tuple,
    and saves the character images into the corresponding directory ('mayus' or 'inus') based on the directory path.
    """
    
    os.mkdir(IMAGES_PATH+f"{new_path}mayus")
    os.mkdir(IMAGES_PATH+f"{new_path}minus")
    for chars,str in characters:
        path=IMAGES_PATH+new_path+str
        path = path.split(image_dir)
        path = path[0]+path[1]
        ff.save_images(chars,path)
        
def __generate_arrays(
    characters: List[tuple[List[MatLike],str]]
) -> tuple[np.ndarray,np.ndarray]:
    """
    This function generates feature and label arrays from a list of character images.

    Parameters:
    -----------
    characters : List[tuple[List[MatLike],str]]
        A list of tuples, where each tuple contains a list of character images and their corresponding directory path.

    Returns:
    -----------
    tuple[np.ndarray,np.ndarray]
        A tuple containing two numpy arrays: the first array represents the features of the characters,
        and the second array represents the corresponding labels.

    Notes:
    -------
    The function iterates over the list of character images, flattens each image into a 1D array,
    and appends it to the 'c' list along with its corresponding label (1-62).
    Finally, it converts the 'c' and 'e' lists into numpy arrays and returns them.
    """
    
    c = [] # Caracteristicas: Array 62 (tipos de caracteres) x 625 (nº imagenes) x 625 (nº pixeles por imagen)
    e = [] # Etiquetas: 1-62
    i=1
    for chars,_ in characters:
        for char in chars:
            c.append(np.ravel(np.array(char)))
            e.append(i)
        i+=1
    return np.array(c),np.array(e)

def __get_arrays(
    image_dir: str,
    new_path: str
) -> tuple[np.ndarray,np.ndarray]:
    """
    This function is responsible for loading images, processing them, and generating arrays for training and testing.

    Parameters:
    -----------
    image_dir : str
        The directory path where the original images are located.
    new_path : str
        The directory path where the processed images will be saved.

    Returns:
    -----------
    tuple[np.ndarray,np.ndarray]
        A tuple containing two numpy arrays: the first array represents the features of the characters,
        and the second array represents the corresponding labels.
    """
    
    #Remover el contenido de las imagenes procesadas:
    if SAVE_PROCESED_IMAGES:
        ff.remove_directory_content(IMAGES_PATH+new_path)
    
    #Cargar las imágenes:
    dirs: List[str] = __get_all_directories(image_dir)
    images_dir: List[tuple[List[MatLike],str]] = __charge_train_images(dirs)
    
    #Procesar las imagenes y obtener los caracteres:
    imgs_preps: List[ImagePreprocessor] = __create_images_preprocessors(images_dir)
    characters: List[tuple[List[MatLike],str]] = [
        (prep.proccess_images(),prep.path) 
        for prep in imgs_preps
    ]
    
    #Guardar las imágenes procesadas:
    if SAVE_PROCESED_IMAGES:
        __save_characters(characters,image_dir,new_path)
    
    #Crear los arrays:
    return __generate_arrays(characters)

def __test_clasifier(
    clf_name:str,
    rd_name:str,
    cr_train: np.ndarray,
    e_train: np.ndarray,
    cr_test: np.ndarray,
    e_test: np.ndarray
) -> str:
    """
    This function is used to test a specific classifier using the given training and testing data.

    Parameters:
    -----------
    clf_name : str
        The name of the classifier to be tested. It should be one of the following:
        'SVC', 'RandomForest', 'KNN', 'LogisticRegression', 'DecisionTree', 'GaussianNB'
    rd_name : str
        The name of dimensionality reducer applied.
    cr_train : np.ndarray
        The transformed training data after applying Linear Discriminant Analysis (LDA).
    e_train : np.ndarray
        The corresponding training labels.
    cr_test : np.ndarray
        The transformed testing data after applying Linear Discriminant Analysis (LDA).
    e_test : np.ndarray
        The corresponding testing labels.

    Returns:
    -----------
    str
        The classification report of the tested classifier.
    """
    
    classifiers = {
        'SVC': SVC(),
        'RandomForest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'DecisionTree': DecisionTreeClassifier(),
        'GaussianNB': GaussianNB()
    }
    clf = classifiers[clf_name]
    clf.fit(cr_train,e_train)
    #Predict:
    pred=clf.predict(cr_test)
    #Metrics:
    mets: str = metrics.classification_report(e_test,pred)
    ff.create_txt(FILES_PATH+rd_name+"_"+clf_name+"_Metrics.txt",mets)
    return mets
    

def exec1(
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
    lda = LinearDiscriminantAnalysis()
    lda.fit(c_train,e_train)
    CR: np.ndarray = lda.transform(c_train)   
    
    CR_TEST: np.ndarray = lda.transform(c_test)
   
    #Test clasifiers:
    if clasifier_name == "All":
        mets = ""
        mets += __test_clasifier('SVC','LDA',CR,e_train,CR_TEST,e_test)
        mets += __test_clasifier('RandomForest','LDA',CR,e_train,CR_TEST,e_test)
        mets += __test_clasifier('KNN','LDA',CR,e_train,CR_TEST,e_test)
        mets += __test_clasifier('LogisticRegression','LDA',CR,e_train,CR_TEST,e_test)
        mets += __test_clasifier('DecisionTree','LDA',CR,e_train,CR_TEST,e_test)
        mets += __test_clasifier('GaussianNB','LDA',CR,e_train,CR_TEST,e_test)
        return mets
    return __test_clasifier(clasifier_name,'LDA',CR,e_train,CR_TEST,e_test)