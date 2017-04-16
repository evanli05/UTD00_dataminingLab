# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:20:57 2017

@author: YL
"""

from detectDrift import calInterval

Mean, lBound, hBound = calInterval('dataCDrift.xlsx')

print lBound