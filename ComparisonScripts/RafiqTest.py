import integrator_class as inty
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Read csv file as a dataframe
data_frame = pd.read_csv('./analysis/data/FILENAME.csv')
print(data_frame)
# number of inputs
D = 7

# read data
param_names = data_frame.keys()[0:D]
#First D colums are the input parameters
inputs = data_frame.values[:, 0:D]
#
mode_frequency = data_frame.values[:, D].reshape([-1,1])
growth_rate = data_frame.values[:, D+1].reshape([-1,1])

tearing = data_frame.values[:, 13].reshape([-1,1])
#Integer indicating whether mode is MTM or not 
mtm = data_frame.values[:, -5]

growths = []
freqs = []
count = 0
iters = 100 #integer determining how many iterations to iterate over

#Reduce arrays to just those iters
mode_frequency = np.array(mode_frequency[:iters])
growth_rate = np.array(growth_rate[:iters])
mtm = np.array(mtm[:iters])

# indices where MTM = 1
idx_conv = np.where(mtm == 1)[0]
# indices where MTM = 0
idx_not_conv = np.where(mtm == 0)[0]

for i in inputs:
    rafiq = inty.calc_omega(ky_in=i[0],q_in=i[1],shat_in=i[2],rln_in=i[3],beta_in=i[4],nu_in=i[5],rlt_in=i[6])
    omega = rafiq.calc()
    count = count + 1
    print(count)
    print(omega.imag, omega.real)
    growths.append(omega.imag)
    freqs.append(omega.real)
    #loop stops after iters number of points
    if count == iters:
        break

growths = np.array(growths)#Convert to numpy arrays then one can mask the array for mtm only points (idx_conv)
freqs = np.array(freqs)

plt.figure(1)
plt.scatter(growth_rate[idx_conv ],growths[idx_conv])
plt.plot(growth_rate,growth_rate,c='red') 
plt.xlabel("GS2 growth rate")
plt.ylabel("Rafiq growth rate")

plt.figure(2)
plt.scatter(mode_frequency[idx_conv],freqs[idx_conv])
plt.plot(mode_frequency,mode_frequency,c='red') 
plt.xlabel("GS2 mode frequency")
plt.ylabel("Rafiq mode frequency")
plt.show()



