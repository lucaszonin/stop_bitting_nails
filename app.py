import numpy as np
import cv2 as cv
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

class PreparingModel(object):

    #Model instancied
    knn = KNeighborsClassifier() 

    def __init__(self):

        self.df_structure = { 
            
            "FILE" : [], # The image 
            "TARGET" : [], # The target (1 = biting nails - 0 = normal)
            "PWD" : [], # The way of the image
            "DESCRIPTION" : [] # Bitting nails or not

        }

        self.loading_dataframe()

    def loading_dataframe(self):

        bitting_nails = os.listdir("imagens/roendo")
        no_bitting_nails = os.listdir("imagens/naoroendo")

        for image in bitting_nails:

            self.df_structure['FILE'].append(self.transforming_image(f"imagens/roendo/{image}"))
            self.df_structure['TARGET'].append(1)
            self.df_structure['PWD'].append(f"imagens/roendo/{image}")
            self.df_structure['DESCRIPTION'].append("bitting_nails")

        for image in no_bitting_nails:

            self.df_structure['FILE'].append(self.transforming_image(f"imagens/naoroendo/{image}"))
            self.df_structure['TARGET'].append(0)
            self.df_structure['PWD'].append(f"imagens/roendo/{image}")
            self.df_structure['DESCRIPTION'].append("no_bitting_nails")

        dataframe = pd.DataFrame(self.df_structure)

        X = list(dataframe["FILE"])
        y = list(dataframe["TARGET"])

        return train_test_split(X, y, test_size=0.3)



    def pca_model(self, X_train):
    
        pca = PCA()

        pca.fit(np.asarray(X_train))

        return pca

    def transforming_image(self, image):

        image_gray = cv.cvtColor(cv.imread(image), cv.COLOR_BGR2GRAY)

        image_transformed = cv.resize(image_gray, dsize=(150,150)).flatten()

        return image_transformed


if __name__ == "__main__":
    d = PreparingModel()    