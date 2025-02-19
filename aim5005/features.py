import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
        
    
    def fit(self, x:np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum=x.min(axis=0)
        self.maximum=x.max(axis=0)
        
    def transform(self, x:np.ndarray) -> list:
        """
        MinMax Scale the given vector
        """
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # TODO: There is a bug here... Look carefully!
        output = [(elem - self.minimum)/(diff_max_min) for elem in x]
        return output
    
    def fit_transform(self, x:list) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std_dv = None
    
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x
    
    def fit(self, x) -> None:
        x = self._check_is_array(x)
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)
    
    def transform(self, x) -> np.ndarray:
        x = self._check_is_array(x)
        output = [(elem - self.mean)/self.std for elem in x]
        return output

class LabelEncoder:
    def __init__(self):
        self.class_dictionary = {}
        self.classes_ = []
    
    def _check_is_array(self, x:np.ndarray) -> np.ndarray:
        """
        Try to convert x to a np.ndarray if it'a not a np.ndarray and return. If it can't be cast raise an error
        """
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a list"
        return x 
    
    def fit(self, x):
        x = self._check_is_array(x)
        count = 0
        x = x.copy()
        x.sort()
        for elem in x:
            if self.class_dictionary.get(elem) == None:
                self.class_dictionary[elem] = count
                count+=1

                #String class
                if isinstance(elem, str):
                    self.classes_.append(str(elem))
                else: #Numeric class
                    self.classes_.append(elem)
        return
    
    def transform(self, x):
        x = self._check_is_array(x)
        output = []
        for elem in x:
            value = self.class_dictionary.get(elem)
            if value == None:
                print(f"Key {elem} not fitted originally!")
                raise KeyError
            output.append(value)
            
        return np.array(output)
    
    def fit_transform(self, x):
        x = self._check_is_array(x)
        self.fit(x)
        output = self.transform(x)
        return output