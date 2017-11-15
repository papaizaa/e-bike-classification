import numpy as np
import csv
from collections import OrderedDict
import matplotlib.pyplot as plt
from textwrap import wrap

# The E-Bike Survey Response Results data set provides us with a CSV formatted survey giving information on Torontonions
# and their knowledge on rules about E-Bike laws within Toronto. The purpose of this exercise is to see if we can
# create a model that can predict whether an individual would answer "No - I do not have access to a private motorized
# vehicle" to the question "Does your household have access to any of the following types of private motorized
# vehicles?".
#
# Here is a Naive Bayes solution to the problem.

########################################################################################################################

# First prepare the data by reducing the multinomial values to integers depending on their first occurrence in
# the data set

filename = 'e-bike-survey.csv'

with open(filename, 'rb') as raw_file:
    raw_data = csv.reader(raw_file, delimiter=',', quoting=csv.QUOTE_NONE)
    data_list = list(raw_data)

ndims = len(data_list[0])
npts = len(data_list)

char_maps = [OrderedDict() for i in range(ndims)]           # Mapping of input responses to integer
reverse_maps = [[] for i in range(ndims)]                   # Array of all the possible values for each field
data_mat = np.empty((npts, ndims), dtype=np.int32)          # Converted matrix of csv
for i, cdata in enumerate(data_list):
    for j, cstr in enumerate(cdata):
        if cstr not in char_maps[j]:
            char_maps[j][cstr] = len(char_maps[j])
            reverse_maps[j].append(cstr)
        data_mat[i, j] = char_maps[j][cstr]
del data_list


np.random.seed(0)
data_perm = np.random.permutation(npts)
data_train = data_mat[data_perm[0:(8*npts/10)], :]          # Use the first 80% of data as training data
data_test = data_mat[data_perm[(8*npts/10):], :]            # Use the last 20% as test data
data_ranges = data_mat[:, 1:].max(axis=0)


########################################################################################################################

# Seperate the data matrix of training data into two distinct matrices.
# One where the result of the question is "No I don't own a motorized..." and the other has an answer of anything else

def generate_labeled_mats(data_train, data_mat):

    vehicle_question_index = 11
    size_list = 2

    labeled_mats = list()

    for i in range(0, size_list):
        if i == 0:
            label_i_array = np.where(data_train[:, vehicle_question_index] == 3)[0]         # Returns a list of indices
        else:
            label_i_array = np.where(data_train[:, vehicle_question_index] != 3)[0]
        len_sub_matrix = len(label_i_array)
        label_i_mat = np.empty((len_sub_matrix, len(data_mat[0])), dtype=np.int32)

        j = 0
        for k in label_i_array:                     # Add elements from the list of indices to a matrix
            label_i_mat[j] = data_train[k]
            j += 1

        labeled_mats.append(label_i_mat)
    return labeled_mats


labeled_mats = generate_labeled_mats(data_train, data_mat)

########################################################################################################################


# Generate prior values for the two outcomes: "No I don't own a motorized vehicle" and other

def generate_priors(labeled_mats, train_data):
    priors = list()
    for i in range(0, len(labeled_mats)):
        prior = float(len(labeled_mats[i]))/len(train_data)
        priors.append(prior)
    return priors

priors = generate_priors(labeled_mats, data_train)

########################################################################################################################


# Plot histograms showing the distribution of feature frequency for both answers to the question
def show_histograms(char_maps, labeled_mats):
    for i in range(1, len(char_maps)):

        if i != 11:
            num_bins = len(char_maps[i]) - 1
            no_car_hist = np.zeros(num_bins)
            yes_car_hist = np.zeros(num_bins)

            for j in range(num_bins):
                no_car_hist[j] = len(np.where(labeled_mats[0][:, i] == j+1)[0])
                yes_car_hist[j] = len(np.where(labeled_mats[1][:, i] == j+1)[0])


            plt.subplot(211)
            plt.ylabel("Number of Occurrences")
            plt.title("\n".join(wrap("Feature {}: {}.\n No Access to Motor Vehicle".format(i,
                                                                                           list(char_maps[i])[0]), 60)))
            plt.bar(range(num_bins), no_car_hist)

            plt.subplot(212)
            plt.ylabel("Number of Occurrences")
            plt.title("\n".join(wrap("Feature {}: {}. \n Other answer".format(i, list(char_maps[i])[0]), 60)))
            plt.xticks(range(num_bins), list(char_maps[i])[1:], rotation='vertical')

            plt.bar(range(num_bins), yes_car_hist)
            # plt.tight_layout()
            plt.show()

# Uncomment this if you would like to see the histograms. Will need to resize the window to see text properly
# Sorry that some graphs are terrible looking, some of the answers to the questions are very long.

# show_histograms(char_maps, labeled_mats)


########################################################################################################################

# For each independent feature distribution and each class label of the feature, calculate the distribution over the
# class label of the answer to the main question: "Does your household have access to any of the following
# types of private motorized vehicles?"
def calculate_likelihoods(a):
    no_vehicle_features = [[] for i in range(ndims)]    # Matrix of posterior values for poisonous features
    yes_vehicle_features = [[] for i in range(ndims)]   # Matrix of posterior values for edible features
    nN = len(labeled_mats[0])                           # Amount of no vehicle values
    yN = len(labeled_mats[1])                           # Amount of yes vehicle values

    for i in range(1, ndims):

        num_bins = len(char_maps[i]) - 1

        ntheta = np.zeros(num_bins)
        ytheta = np.zeros(num_bins)

        for j in range(num_bins):
            no_amount = len(np.where(labeled_mats[0][:, i] == j+1)[0])
            yes_amount = len(np.where(labeled_mats[1][:, i] == j+1)[0])

            # Use the Dirichlet distribution where all hyper parameters will equal to `a`
            # Derive the likelihoods using the MAP Estimate for a Dirichlet distribution
            ytheta[j] = (float(yes_amount) + a - 1)/(yN + num_bins * a - num_bins)
            ntheta[j] = (float(no_amount) + a - 1)/(nN + num_bins * a - num_bins)

        no_vehicle_features[i-1] = ntheta
        yes_vehicle_features[i-1] = ytheta
    return no_vehicle_features, yes_vehicle_features


# Function to find the posterior value for each piece of data in the dataset.
# Because of issues with finite precision in a computer, take advantage of the log sum exp trick.
# This trick ensures data is in the same magnitude and is less likely to suffer from rounding errors
def log_sum_exp(no_vehicle_features, yes_vehicle_features, data):

    amount_correct = 0

    for i in range(0, len(data)):

            lp1 = 0
            lp0 = 0
            for j in range(1, len(data[0])):

                if j != 11:                                         # Ignore j = 11, since this is our class label
                    feature_val = data[i][j] - 1

                    if yes_vehicle_features[j-1][feature_val] != 0:
                        lp1 += np.log(yes_vehicle_features[j-1][feature_val])

                    if no_vehicle_features[j-1][feature_val] != 0:
                        lp0 += np.log(no_vehicle_features[j-1][feature_val])

            lp1 += np.log(priors[1])
            lp0 += np.log(priors[0])
            B = max(lp1, lp0)
            posterior = lp1 - (np.log(np.exp(lp1 - B) + np.exp(lp0 - B)) + B)

            if posterior > np.log(0.5):
                if data[i][11] != 3:
                    amount_correct += 1
            if posterior <= np.log(0.5):
                if data[i][11] == 3:
                    amount_correct += 1

    return amount_correct


alpha = np.arange(1, 2.01, 0.001)


# Over a range of Dirichlet hyper parameters (alpha), find the prediction accuracy achieved by each value of a
def find_prediction_accuracy_for_alpha_range(alpha, data):
    best = -1000000
    best_a = -1000000
    all_alpha = np.zeros(len(alpha))
    size_data = len(data)
    i = 0
    for a in alpha:
        no_vehicle_features, yes_vehicle_features = calculate_likelihoods(a)
        amount_accuracy = float(log_sum_exp(no_vehicle_features, yes_vehicle_features, data))/size_data
        all_alpha[i] = amount_accuracy
        i += 1
        if amount_accuracy > best:
            best = amount_accuracy
            best_a = a
    print "Best accuracy was {}%".format(best*100)
    return best_a, all_alpha

best_a_train, all_alpha_train = find_prediction_accuracy_for_alpha_range(alpha, data_train)
print "The best alpha value for training data is {}".format(best_a_train)

best_a_test, all_alpha_test = find_prediction_accuracy_for_alpha_range(alpha, data_test)
print "The best alpha value for test data is {}".format(best_a_test)


plt.xlabel("Alpha values")
plt.ylabel("Probability")
plt.plot(alpha, all_alpha_train, label="Train Set")
plt.plot(alpha, all_alpha_test, label="Validation Set")
plt.legend()
plt.show()


########################################################################################################################


