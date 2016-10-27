import csv as csv 
import numpy as np

# Open up the csv file in to a Python object
csv_file_object = csv.reader(open('data/train.csv', 'rb')) 
header = csv_file_object.next()  # The next() command just skips the 
                                 # first line which is a header
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)             # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
						         # Be aware that each item is currently
                                 # a string in this format


# The size() function counts how many elements are in
# in the array and sum() (as you would expects) sums up
# the elements in the array.

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

# print proportion_survivors

women_only_stats = data[0::,4] == "female" # This finds where all 
                                           # the elements in the gender
                                           # column that equals female
men_only_stats = data[0::,4] != "female"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)


# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

# test_file = open('data/test.csv', 'rb')
# test_file_object = csv.reader(test_file)
# header = test_file_object.next()

# prediction_file = open("data/genderbasedmodel.csv", "wb")
# prediction_file_object = csv.writer(prediction_file)

# prediction_file_object.writerow(["PassengerId", "Survived"])
# for row in test_file_object:       # For each row in test.csv
#     if row[3] == 'female':         # is it a female, if yes then                                       
#         prediction_file_object.writerow([row[0],'1'])    # predict 1
#     else:                              # or else if male,       
#         prediction_file_object.writerow([row[0],'0'])    # predict 0
# test_file.close()
# prediction_file.close()

# In order to analyse the price column I need to bin up that data
# here are my binning parameters, the problem we face is some of the fares are very large
# So we can either have a lot of bins with nothing in them or we can just lose some
# information by just considering that anythng over 39 is simply in the last bin.
# So we add a ceiling
fare_ceiling = 40
# then modify the data in the Fare column to = 39, if it is greater or equal to the ceiling
data[ data[0::,9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling / fare_bracket_size
number_of_classes = 3                             # I know there were 1st, 2nd and 3rd classes on board.
number_of_classes = len(np.unique(data[0::,2]))   # But it's better practice to calculate this from the Pclass directly:
                                                  # just take the length of an array of UNIQUE values in column index 2


# This reference matrix will show the proportion of survivors as a sorted table of
# gender, class and ticket fare.
# First initialize it with all zeros
survival_table = np.zeros([2,number_of_classes,number_of_price_brackets],float)


for i in xrange(number_of_classes):
    for j in xrange(number_of_price_brackets):
    	
        women_only_stats = data[ (data[0::,4] == "female") #Where data is female
                                 & (data[0::,2].astype(np.float) == i+1) #Where equals to current class i.e., 1,2,3
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) # 
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

		 
        men_only_stats = data[ (data[0::,4] != "female") 
                                 & (data[0::,2].astype(np.float) == i+1) 
                                 & (data[0:,9].astype(np.float) >= j*fare_bracket_size) 
                                 & (data[0:,9].astype(np.float) < (j+1)*fare_bracket_size), 1]

                                 #if i == 0 and j == 3:

        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))  # Female stats
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))    # Male stats

survival_table = np.nan_to_num(survival_table)


# Now I have my proportion of survivors, simply round them such that if <0.5
# I predict they dont surivive, and if >= 0.5 they do
survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1

