Each pkl file contains coordinates of monitored and unmonitored locations for Catalonia Air pollution network.
The indices are sorted with idx 0 being Ciutadella location. As the index increases the location is further away from Ciutadella station.
Normal IRNet algorithm parameters include epsilon=1e-2 and n_it = 50. 
As epsilon decreases more locations are monitored.
If using large epsilon (~1e-1) and low threshold ratios (~1.1) the solution might not fullfill the required threshold design.
If using low epsilon (~1e-2) and large threshold ratios (~1.4) the algorithm might select too many locations.

Parameters:
N=48,s=36
---------

1) Distance-based ROIs: BCN and rural.
1.1) 2ROIs: 10km thresholds from Ciutadella station. 8 and 40 elements per ROI
---------------------------------------------------------------------------
rho,		epsilon,	p,	t,		fully_monitored,	threshold,		deployed
[1.01,1.5],	1e-2,		46,	598.76,		[0.896,0.995],		[0.905,1.493],		[0.900,1.336]
[1.05,1.5],	1e-2,		46,	452.64,		[0.896,0.995],		[0.941,1.493],		[0.908,1.365]
[1.1,1.5],	5e-3,		40,	228.81,		[0.896,0.995],		[0.985,1.493],		[0.962,1.459]
[1.2,1.5],	5e-3,		40,	167.31,		[0.896,0.995],		[1.075,1.493],		[0.962,1.459]
[1.3,1.5],	5e-3,		40,	161.58,		[0.896,0.995],		[1.165,1.493],		[0.962,1.459]
[1.4,1.5],	5e-3,		40,	175.68,		[0.896,0.995],		[1.254,1.493],		[0.962,1.459]
[1.1,1.3],	5e-3,		41,	265.34,		[0.896,0.995],		[0.986,1.294],		[0.937,1.234]
[1.01,1.3],	1e-2,		45,	590.83,		[0.896,0.995],		[0.905,1.294],		[0.900,1.125]

----------------------------------------------------------------------------

2) Variance-based ROIs.
2.1) 2ROIs:  0.75 threshold. 23 and 25 elements per ROI
-----------------------------------------------------
rho		epsilon		p	t,		fully_monitored,	threshold,	deployed
[1.4,1.5]	1e-2		47	474.04		[0.708,0.995]		[0.991,1.493]	[0.932,0.995]
[1.3,1.5]	1e-1		46	185.95		[0.708,0.995]		[0.921,1.493]	[0.915,0.995]
[1.2,1.5]	5e-2		47	380.53		[0.708,0.995]		[0.849,1.493]	[0.765,0.995]
[1.1,1.5]	5e-2		47	439.84		[0.708,0.995]		[0.779,1.493]	[0.708,0.995]
[1.05,1.5]	1e-2		47	439.55		[0.708,0.995]		[0.744,1.493]	[0.708,0.995]
[1.01,1.5]	1e-2		47	604.46		[0.708,0.995]		[0.715,1.493]	[0.708,0.995]
[1.1,1.3]	1e-2		47	546.73		[0.708,0.995]		[0.779,1.294]	[0.708,0.995]
[1.01,1.3]	1e-2		47	599.81		[0.708,0.995]		[0.715,1.294]	[0.708,0.995]
[1.5,2.0]	2e-1		44	190.26		[0.708,0.995]		[1.062,1.991]	[1.015,0.996]
[2.0,2.5]	2e-1		40	147.36		[0.708,0.995]		[1.416,2.489]	[1.250,0.999]
---------------------------------------------------------

2.2) 3ROIs: 0.5 and 0.75 threshold. 5, 18, 25 locations per ROI.
----------------------------------------------------------------
rho		epsilon	p	t	fully_monitored		threshold		deployed
[1.3,1.5,2.0]	1e-1	46	335.58	[0.495,0.708,0.995]	[0.643,1.062,1.991]	[0.592,1.005,0.995]
[1.5,2.0,2.5]	5e-2	43	219.71	[0.495,0.708,0.995]	[0.742,1.416,2.489]	[0.635,1.233,0.999]
