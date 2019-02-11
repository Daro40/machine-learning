import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

# average 24 and 28 inches, normal distribution +/-4 inches
grey_height = 28 + 4*np.random.randn(greyhounds)
lab_height = 24 + 4*np.random.randn(labs)

#plot
plt.hist([grey_height, lab_height], stacked= True, color =['r', 'g'])
plt.show()