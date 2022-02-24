import pandas as pd
from project_functions import *
import matplotlib.pyplot as plt



################ Initial Data Wrangling ################
"""
Set up column names for data, since they're not included in the original data set. For the quantitative data values, they are given descriptive names.
In some cases, I included abbreviations
    HD = Horizontal Distance To <feature>
    VD = Vertical Distance to <feature>

The one-hot encoded categorical data was given an abbreviated prefix and a numerical suffix. 
In this case "WA" = Wilderness Area and "ST" = Soil Type. So the first kind of soil will have a column name of "ST_1".
"""

column_names = ["Elevation", "Aspect", "Slope", "HD_Hydrology", "VD_Hydrology", "HD_Roadways", "Shade_9am", "Shade_12pm", "Shade_3pm", "HD_FirePoints"]

for i in range(4):
    column_names.append("WA_" + str(i+1))

for i in range (0,40):
    column_names.append("ST_" + str(i+1))

column_names.append("CoverType")

# take first thousands instances of the read data, set column names as defined above
file_path = "./data/covtype.data"
cover_data_pd = pd.read_csv(file_path, delimiter=",", nrows=1000, names=column_names)

# put into 2d numpy array (as opposed to pandas dataframe) so it works with the functions we were supposed to write
cover_data = cover_data_pd.to_numpy()


################ Multidimensional Mean ################
multidimensional_mean = get_multidimensional_mean(cover_data)
print("Multidimensional mean: \n", multidimensional_mean, "\n\n")


################ Covariance Matrix ################
covariance_matrix = get_covariance_matrix(cover_data)
print("Covariance matrix: \n", covariance_matrix, "\n\n")


################ Scatter Plots ################
plt.scatter(cover_data_pd["Elevation"], cover_data_pd["Slope"], c="blue")
plt.title("Elevation by Slope")
plt.xlabel("Elevation (meters)")
plt.ylabel("Slope (degrees)")

plt.savefig("./plots/figure1.png")
plt.clf()


plt.scatter(cover_data_pd["HD_Roadways"], cover_data_pd["Elevation"],c="orange")
plt.title("Distance to Roadways by Elevation")
plt.xlabel("Horizontal Distance (meters)")
plt.ylabel("Elevation (meters)")

plt.savefig("./plots/figure2.png")
plt.clf()


plt.scatter(cover_data_pd["HD_Hydrology"], cover_data_pd["VD_Hydrology"],c="green")
plt.title("Comparison of Distance to Water Features")
plt.xlabel("Horizontal Distance (meters)")
plt.ylabel("Vertical Distance (meters)")

plt.savefig("./plots/figure3.png")
plt.clf()


plt.scatter(cover_data_pd["Shade_9am"], cover_data_pd["Shade_3pm"],c="red") 
plt.title("Morning vs Afternoon Hillshade")
plt.xlabel("Shade 9am (index 0-255)")
plt.ylabel("Shade 3pm (index 0-255)")

plt.savefig("./plots/figure4.png")
plt.clf()

plt.scatter(cover_data_pd["Elevation"], cover_data_pd["Shade_12pm"], c="purple")
plt.title("Mid-Day Shade at Differen Elevations")
plt.xlabel("Elevation (meters)")
plt.ylabel("Shade 12pm (index 0-255)")

plt.savefig("./plots/figure5.png")
plt.clf()


################ Range Normalization and Covaraince Analysis ################

# normalize and get covariance matrix
range_normalized_data = get_range_normalization(cover_data)
range_normalized_covariance_matrix = get_covariance_matrix(range_normalized_data)

# set up tracking variables
covariance_attributes = [0,0]
negative_covariance_pairs = dict()
max_covariance = -100000

# iterate over covariance matrix (not a super effecient method but it'll send)
for i in range(range_normalized_covariance_matrix.shape[1]):
    for j in range(range_normalized_covariance_matrix.shape[0]):
        # we don't need to compare any column to itselft
        if i != j:
            # get covariance of the two attributes
            cur_covariance = range_normalized_covariance_matrix[j,i]

            # update max covariance if neccessary
            if cur_covariance > max_covariance:
                covariance_attributes = [i,j]
                max_covariance = cur_covariance
            
            # track negative covariance pairs
            if cur_covariance < 0:
                negative_covariance_pairs.update({i:j})
            

# get column names of the two attributes with the largest covariance
covariance_attribute_1 = column_names[covariance_attributes[0]]
covariance_attribute_2 = column_names[covariance_attributes[1]]

print("Largest covariance: ", max_covariance)
print("Attributes:\n\t",covariance_attribute_1,"\n\t",covariance_attribute_2)

print("Number of negative covariance pairs", len(negative_covariance_pairs),"\n\n")

# create scatter plot for attributes with greatest covariances
plt.scatter(cover_data_pd[covariance_attribute_1], cover_data_pd[covariance_attribute_2], c="blue")
plt.title("Attributes with Greatest Covariance")
plt.ylabel(covariance_attribute_1)
plt.xlabel(covariance_attribute_2)

plt.savefig("./plots/greatest_covariance_plot.png")
plt.clf()


################ Standard Normalization and Correlation Analysis ################

# get standard normalization of the data
z_normalized_data = get_standard_normalization(cover_data)

# set up tracking variables
max_correlation = -100000
correlation_attributes_max = [0,0]

min_correlation = 100000
correlation_attributes_min = [0,0]

correlation_feature_pairs = dict()

# iterate over normalized data
for i in range(z_normalized_data.shape[1]):
    for j in range(z_normalized_data.shape[1]):
        # again, no need to compare a column to itself
        if i != j:
            # get the correlation of two non-same columns
            cur_correlation = get_correlation(z_normalized_data[:,j], z_normalized_data[:,i])

            # update max correlation, if necessary
            # note the absolute value function, since "low" negative correlations still have a high correlation
            #       (i.e a correlation of -0.9 is still greater than, say, a correlation of 0.2)
            if abs(cur_correlation) > max_correlation:
                max_correlation = cur_correlation
                correlation_attributes_max = [i,j]
            
            # update min correlation, if necessary
            if abs(cur_correlation) < min_correlation:
                min_correlation = cur_correlation
                correlation_attributes_min = [i,j]

            # track correlations that are greater than or equal to 0.5
            # only add if the pair isn't already in our tracking dictionary
            if abs(cur_correlation) >= 0.5 and not j in correlation_feature_pairs:
                correlation_feature_pairs.update({i : j})


max_cor_attribute_1 = column_names[correlation_attributes_max[0]]
max_cor_attribute_2 = column_names[correlation_attributes_max[1]]

min_cor_attribute_1 = column_names[correlation_attributes_min[0]]
min_cor_attribute_2 = column_names[correlation_attributes_min[1]]

print("Largest correlation: ", max_correlation)
print("Attributes:\n\t",max_cor_attribute_1,"\n\t",max_cor_attribute_2)

print("Smallest correlation: ", min_correlation)
print("Attributes:\n\t",min_cor_attribute_1,"\n\t",min_cor_attribute_2)

print("Number of feature pairs w/ correlation >= 0.5: ", len(correlation_feature_pairs), "\n\n")


# create scatter plot of attributes with greatest correlation
plt.scatter(cover_data_pd[max_cor_attribute_1], cover_data_pd[max_cor_attribute_2], c="orange")
plt.title("Attributes with Greatest Correlation")
plt.xlabel(max_cor_attribute_1)
plt.ylabel(max_cor_attribute_2)

plt.savefig("./plots/greatest_correlation_plot.png")
plt.clf()

# create scatter plot of attributes with smallest correlation
plt.scatter(cover_data_pd[min_cor_attribute_1], cover_data_pd[min_cor_attribute_2], c="purple")
plt.title("Attributes with Smallest Correlation")
plt.xlabel(min_cor_attribute_1)
plt.ylabel(min_cor_attribute_2)

plt.savefig("./plots/smallest_correlation_plot.png")
plt.clf()


################ Total variances ################

# set up tracking variables
total_variance = 0
greatest_variances = []

# iterate over columns in data
for i in range(cover_data.shape[1]):
    # get the current column's variance and add to the total variance
    cur_variance = round(get_variance(cover_data[:,i]),2)
    total_variance += cur_variance

    # if greatest variances hasn't been filled with the five largest values - simply append the current value
    if len(greatest_variances) < 5:
        greatest_variances.append(cur_variance)
    else:
        # if the greatest variances is filled, check to see if the current variance is greatest than the smallest value in the array
        #   and update if necessary
        smallest_variance = min(greatest_variances)
        
        if smallest_variance < cur_variance:
            index_val = greatest_variances.index(smallest_variance)
            greatest_variances[index_val] = cur_variance

# round the total variance
total_variance = round(total_variance,2)

# calculate total variance of the largest values and round
total_variance_large = round(sum(greatest_variances),2)


print("Total variance: ", total_variance)
print("Large total variance: ", total_variance_large)