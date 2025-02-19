from aim5005.features import MinMaxScaler, StandardScaler, LabelEncoder
import numpy as np
import unittest
from unittest.case import TestCase

### TO NOT MODIFY EXISTING TESTS

class TestFeatures(TestCase):
    def test_initialize_min_max_scaler(self):
        scaler = MinMaxScaler()
        assert isinstance(scaler, MinMaxScaler), "scaler is not a MinMaxScaler object"
        print(f"Passed=test_initialize_min_max_scaler")
        
        
    def test_min_max_fit(self):
        scaler = MinMaxScaler()
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        scaler.fit(data)
        assert (scaler.maximum == np.array([1., 18.])).all(), "scaler fit does not return maximum values [1., 18.] "
        assert (scaler.minimum == np.array([-1., 2.])).all(), "scaler fit does not return maximum values [-1., 2.] " 
        print(f"Passed=test_min_max_fit")
        
    def test_min_max_scaler(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. All Values should be between 0 and 1. Got: {}".format(result.reshape(1,-1))
        print(f"Passed=test_min_max_scaler")

    def test_min_max_scaler_single_value(self):
        data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
        expected = np.array([[1.5, 0.]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect [[1.5 0. ]]. Got: {}".format(result)
        print(f"Passed=test_min_max_scaler_single_value")
        
    def test_standard_scaler_init(self):
        scaler = StandardScaler()
        assert isinstance(scaler, StandardScaler), "scaler is not a StandardScaler object"
        print(f"Passed=test_standard_scaler_init")
        
    def test_standard_scaler_get_mean(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([0.5, 0.5])
        scaler.fit(data)
        assert (scaler.mean == expected).all(), "scaler fit does not return expected mean {}. Got {}".format(expected, scaler.mean)
        print(f"Passed=test_standard_scaler_get_mean")

    def test_standard_scaler_transform(self):
        scaler = StandardScaler()
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[-1., -1.], [-1., -1.], [1., 1.], [1., 1.]])
        scaler.fit(data)
        result = scaler.transform(data)
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        print(f"Passed=test_standard_scaler_transform")

    def test_standard_scaler_single_value(self):
        data = [[0, 0], [0, 0], [1, 1], [1, 1]]
        expected = np.array([[3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        print(f"Passed=test_standard_scaler_single_value")
    
    # TODO: Add a test of your own below this line
    def test_standard_scaler_3d_array(self):
        data = [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1,1 , 1]]
        expected = np.array([[3., 3., 3.]])
        scaler = StandardScaler()
        scaler.fit(data)
        result = scaler.transform([[2., 2., 2.]]) 
        assert (result == expected).all(), "Scaler transform does not return expected values. Expect {}. Got: {}".format(expected.reshape(1,-1), result.reshape(1,-1))
        print(f"Passed=test_standard_scaler_3d_array")

    def test_label_encoder_class_initialization(self):
        encoder = LabelEncoder()
        assert isinstance(encoder, LabelEncoder), "Encoder is not a LabelEncoder object"
        print(f"Passed=test_label_encoder_class_initialization")
    
    def test_label_encoder_classes(self):
        scaler = LabelEncoder()
        data = ["paris", "paris", "tokyo", "amsterdam"]
        scaler.fit(data)
        expected = ['amsterdam', 'paris', 'tokyo']

        assert (scaler.classes_ == expected), "LabelEncoder did not return expected classes. Expect {}. Got: {}".format(expected, scaler.classes_)
        print(f"Passed=test_label_encoder_classes")

    def test_label_encoder_transform(self):
        scaler = LabelEncoder()
        fit_data = ["paris", "paris", "tokyo", "amsterdam"]
        scaler.fit(fit_data)

        data = ["tokyo", "tokyo", "paris"]
        expected = [2, 2, 1]

        result = scaler.transform(data)
        assert (result == expected).any(), "LabelEncoder did not return expected transform labels. Expect {}. Got: {}".format(expected, result)
        print(f"Passed=test_label_encoder_transform")

    def test_label_encoder_numeric_classes(self):
        scaler = LabelEncoder()
        fit_data = [1, 2, 2, 6]
        scaler.fit(fit_data)

        expected = [1, 2, 6]

        assert (scaler.classes_ == expected), "LabelEncoder did not return expected classes for numeric labels. Expect {}. Got: {}".format(expected, scaler.classes_)
        print(f"Passed=test_label_encoder_numeric_classes")

    def test_label_encoder_numeric_classes_transform(self):
        scaler = LabelEncoder()
        fit_data = [1, 2, 2, 6]
        scaler.fit(fit_data)

        expected = [0, 0, 1, 2]
        result = scaler.transform([1, 1, 2, 6])
        assert (result == expected).all(), "LabelEncoder did not return expected output after transform. Expect {}. Got: {}".format(expected, result)
        print(f"Passed=test_label_encoder_numeric_classes_transform")

if __name__ == '__main__':
    unittest.main()