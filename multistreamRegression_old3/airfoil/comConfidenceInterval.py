# -*- coding: utf-8 -*-

"""
Created on Tue Apr 11 20:00:57 2017

@author: YL
"""

import xlrd
import numpy as np

setName = 'dataCDrift.xlsx'

workbook = xlrd.open_workbook(setName)
sheet  = workbook.sheet_by_index(0)
#print sheet.nrows

data = [[sheet.cell_value(r,c) for c in range(sheet.ncols)] for r in \
         range(sheet.nrows)]

#print data[0]
npdata = np.array(data)                                 
                                   