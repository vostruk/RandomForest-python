
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from numpy import genfromtxt

import sys
#sys.argv[0]
my_data = genfromtxt('madelon_trees_res.txt', delimiter=',')

with PdfPages('madelon_chart.pdf') as pdf:
    plt.figure(1)
    plt.subplot(211)
    plt.plot(my_data[:,0], my_data[:,1], 'r--', my_data[:,0], my_data[:,2], 'b--')
    plt.title('Madeon')
    plt.xlabel('Number of trees')
    plt.ylabel('F1Score')
    pdf.savefig()
    plt.close()
