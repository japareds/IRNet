Each pkl file contains coordinates of monitored and unmonitored locations of Catalonia Air pollution network.
The indices are sorted with idx 0 being Ciutadella location. As the index increase the location is further away from Ciutadella station.
Normal IRNet parameters include epsilon = 1e-2 and n_it = 50. 
As epsilon decreases it tends to select more sensors.
If using epsilon = 1e-1 for low threshold ratios (~1.1) the IRNet algorithm solution does not fullfill the threshold design.
If using low epsilon (1e-2) for large threshold ratios (~1.4) the IRNet algorithm tends to select multiple locations. An alternative is to increase epsilon in order to reduce the number of locations.

Parameters:
----------
N=48,s=36
---------
rho,	epsilon,	p,	t,		fully_monitored,	threshold,	deployed
1.01,	5e-3,		45,	825.65,		0.995,			1.005,		0.997
1.05,	5e-3,		44,	467.94,		0.995,			1.045,		1.015
1.1,	5e-3,		43,	336.86,		0.995,			1.095,		1.089
1.2,	1e-2,		42,	222.35,		0.995,			1.195,		1.148
1.3,	5e-2,		41,	207.42,		0.995,			1.294,		1.235
1.4,	2e-2,		41,	179.71,		0.995,			1.394,		1.235
1.5,	2e-2,		41,	202.17,		0.995,			1.493,		1.235

---------
N=48,s=30
---------
rho,epsilon,p
1.5,5e-2,33
1.4,5e-2,33
1.3,1e-2,35
1.2,1e-2,35
1.1,1e-2,36
1.05,1e-2,36
1.01,1e-3,37


