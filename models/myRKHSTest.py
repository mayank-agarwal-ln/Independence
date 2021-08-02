# ----------------------------------------------------------------------------
# Model Definition
# ----------------------------------------------------------------------------

# Describe the test
testDescript = 'RKHS Test Data'

# Define the causal model.
# Each random variable has the following fields:
# - Name
# - List of Parents
# - isObserved (Optional, default True)
# - Data Type (Optional, default 'Numeric')  
model =    ['X'
			]

varEquations = ['X = logistic(0,1)']
# varEquations = ['X = exponential(0.5) if choice([0,1]) else logistic(2,2)']
# varEquations = ['X = exponential()']