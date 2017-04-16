# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:26:13 2017

@author: yl
"""

import matlab.engine
eng = matlab.engine.start_matlab()
eng.pythonTest(nargout=0)