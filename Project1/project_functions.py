import numpy as np
import math

# scientific notation is annoying
np.set_printoptions(suppress=True)

# function to get mean of a numpy array
def get_column_mean(column):
    sum = 0
    for i in column:
        sum += i

    return sum/len(column)

# utilizes the column mean function to calculate the multidimensional mean of a 2d numpy array
def get_multidimensional_mean(matrix_array):
    return_array = [0]*matrix_array.shape[1]

    for i in range(matrix_array.shape[1]):
        return_array[i] = round(get_column_mean(matrix_array[:,i]),2) 

    return return_array

# get variance of numpy array
def get_variance(vector_array):
    variance = 0

    # get mean of array
    series_mean = get_column_mean(vector_array)
    
    # loop over array and calculate running sum
    for i in vector_array:
        variance += (i - series_mean)**2
        
    # calculate final variance    
    return variance/ (len(vector_array) - 1)


# get covariance of two numpy arrays
def get_covariance(vector_array_1, vector_array_2):
    # get the mean of each array
    vector_1_mean = get_column_mean(vector_array_1)
    vector_2_mean = get_column_mean(vector_array_2)
    sum_val = 0

    # iterate over and calculate runnign sum
    for i in range(len(vector_array_1)):
        sum_val += ((vector_array_1[i] - vector_1_mean) * (vector_array_2[i] - vector_2_mean))

    # calculate and round final covairance value
    return round(sum_val/(len(vector_array_1) - 1),2)


# get standard deviation of a numpy array
def get_standard_deviation(vector_array):
    # get variance and return its square root
    variance = get_variance(vector_array)
    return math.sqrt(variance)


# get correlation of two numpy arrays
def get_correlation(vector_array_1, vector_array_2):
    # get covariance of the two arrays
    vector_covariance = get_covariance(vector_array_1, vector_array_2)

    # get the standard deviation of each array
    vector_1_stdev = get_standard_deviation(vector_array_1)
    vector_2_stdev = get_standard_deviation(vector_array_2)

    # finally calculate the correlation
    return vector_covariance / (vector_1_stdev * vector_2_stdev)


# function to apply range normalization to a 2d numpy array
def get_range_normalization(np_array):
    # get number of ros and columns in the matrix
    n_rows = np_array.shape[0]
    n_cols = np_array.shape[1]

    # create new 2d numpy array to return
    normalized_data = np.empty(shape=(n_rows, n_cols))
    normalized_data.fill(0)

    for i in range(0, n_cols):
        # get the min and max values in the current column
        col_min = min(np_array[:,i])
        col_max = max(np_array[:,i])

        for j in range(0, n_rows):
            # normalize each value in the current column
            normalized_data[j,i] = round((np_array[j,i] - col_min) / (col_max - col_min),2)

    return normalized_data

# function to apply standard(z-score) normalization to a 2d numpy array
def get_standard_normalization(np_array):
    n_rows = np_array.shape[0]
    n_cols = np_array.shape[1]

    normalized_data = np.empty(shape=(n_rows, n_cols))
    normalized_data.fill(0)

    for i in range(n_cols):
        # get mean and standard deviation for the current column in the array
        col_mean = get_column_mean(np_array[:,i])
        col_stdev = get_standard_deviation(np_array[:,i])

        for j in range(n_rows):
            # apply normalization to each value in the current column
            normalized_data[j,i] = round((np_array[j,i] - col_mean) / (col_stdev), 2)

    
    return normalized_data


# function to get the covariance matrix of a 2d numpy array
def get_covariance_matrix(np_array):
    # get number of columns
    n_cols = np_array.shape[1]

    # create return array to hold covariance matrix
    covariance_matrix = np.empty(shape=(n_cols, n_cols))
    covariance_matrix.fill(0)

    # iterate over every column twice (a disgusting O(2n) algorithm that could definitely be simplified)
    # compare each column to every other one in the array and calculate their covairance
    # additionally, calculate the variance for each column
    for i in range(n_cols):
        col_variance = get_variance(np_array[:,i])
        covariance_matrix[i,i] = round(col_variance,2)

        for j in range(n_cols):
            if j != i:
                np_sub_1 = np_array[:,i]
                np_sub_2 = np_array[:,j]
                cur_covariance  = get_covariance(np_sub_1, np_sub_2)
                covariance_matrix[j,i] = round(cur_covariance,2)
        
        
    return covariance_matrix


# function to label encode a 2d numpy array
def get_label_encoded(np_array):
    # get number of rows and columns
    n_rows = np_array.shape[0]
    n_cols = np_array.shape[1]

    # create new numpy array to return
    label_encoded_array = np.empty(shape=(np_array.shape))
    label_encoded_array.fill(0)

    # iterate over every column
    for i in range(n_cols):
        # get the indexes of the unique values in the current column
        indexes = np.unique(np_array[:,i], return_index=True)[1]

        # return the unique values in the current column in the order they appear
        # we could just use unique() directly on the column, but this will order the values
        #    and mess up the label encoding 
        unique_vals = np.asarray([np_array[:,i][index] for index in sorted(indexes)])

        # iterate over the values in the column and apply the label encoding
        for j in range(n_rows):
            val_to_set = np.where(unique_vals == np_array[j,i])
            label_encoded_array[j,i] = val_to_set[0][0]


    return label_encoded_array