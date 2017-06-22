class DataPoint:
	def __init__(self, indVars, y):
		self.indVars = indVars
		self.y = y

	def toString(self):
		strPoint = ""
		for i in self.indVars:
			strPoint += (str(i) + ",")
		return strPoint + str(self.y) + "\n"