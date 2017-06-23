import numpy as np 
import matplotlib.pyplot as plt


fig, ax = plt.subplots()

matrix = np.genfromtxt('C:/Users/skinnycook/OneDrive - The University of Texas at Dallas/Workspace/UTD00_dataminingLab/multistreamRegression/RegSynGlobalAbruptDrift_target_stream.csv, delimiter=',')
plt.plot(data[:,5], data[:,0], 'ro') 
plt.show()

x = [9000, 18000, 27000, 36000, 45000, 54000, 63000, 72000, 81000, 90000]
RCSR = [2.50245883738, 2.41639560221, 14.1541019904, 12.9724696344, 12.28889386, 12.0164437225, 11.7618856395, 11.4367209141, 11.1665411437, 10.9084725959]
pRCSR = [2.41845365131613, 2.41845365131613, 3.417596543827754, 5.6580205337055345, 5.6580205337055345, 5.6580205337055345, 5.419971337293677, 5.419971337293677, 5.18813018250524, 5.18813018250524]
EMR = [2.495714117875685, 2.401398344799003, 4.10262561801443, 4.802005849452552, 5.05994113755002, 4.781684867880891, 4.6237441852501675, 4.605728051574869, 4.89646498102114, 4.89646498102114]

ax.plot(x, RCSR, 'b--', label='RCSR', marker='*')
ax.plot(x, pRCSR, 'g--', label='pRCSR', marker='*')
ax.plot(x, EMR, 'r--', label='EMR', marker='o')

ax.set_ylim([-2,15])
ax.set_xlabel('Data Sequence')
ax.set_ylabel('Average Error of Target Dataset')
ax.set_title('Local Abrupt Drift Dataset')

plt.legend(loc=3, borderaxespad=0.)

plt.show()