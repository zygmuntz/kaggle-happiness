'bag predictions from imputed data sets'

import numpy as np
from glob import glob

input_pattern = '/path/to/many/predictions/dir/*.csv'

output_file = '/path/to/where/you/want/p_imputed_bagged.csv'

# in case we re-run the script in ipython
p = None

files = glob( input_pattern )
for input_file in files:
	print input_file
	
	data = np.loadtxt( input_file, skiprows = 1, delimiter = ',' )
	try:
		p = np.hstack(( p, data[:,1:2] ))
	except ValueError:
		# the first file
		p = data[:,1:2]
		ids = data[:,0:1]
		
print p.shape

# average
p = np.mean( p, axis = 1 ).reshape( -1, 1 )

ids_and_p = np.hstack(( ids, p ))
np.savetxt( output_file, ids_and_p, fmt = [ '%d', '%.10f' ], delimiter = ',', header = 'UserID,Probability1', comments = '' )
