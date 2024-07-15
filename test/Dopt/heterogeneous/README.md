D-optimalty sensor placement results for network splitted into different Regions of Interest (ROIs)
The ROIs are defined by the Network design algorithm. D-opt is just for comparison.
D-opt requires knowing the number of sensors deployed in each ROI according to the Network Design algorithm.

Parameters:
N=48,s=36

1) distance-based ROIs: BCN and rural
1.1) 2 ROIs: BCN and rural. 10km threshold from Ciutadella station. 8 and 40 locations per ROI

The first reported row for each design threshold corresponds to D-opt mimic Network design algorithm.
The last row for each design threshold corresponds to the minimum number of sensors possible.
------------------------------------------------------------------------------------------------
design threshold,	n_sensors,	n_sensors/ROI,	Dopt-performance,	figure
[1.3,1.5],		40,		[7,33],			bad		yes
[1.3,1.5],		40,		[8,32],			good		yes
[1.3,1.5],		40		[6,34],
[1.3,1.5],		40,		[5,35],			bad	
[1.3,1.5],		41		[

[1.3,1.5],		36,		[8,28],			bad
[1.3,1.5],		36,		[5,31],			bad


[1.1,1.5],		40,		[7,33],			good
[1.1,1.5],		40,		[8,32],			good
[1.1,1.5],		36,		[8,28],			bad
[1.1,1.5],		36,		[5,31],			bad

[1.01,1.5],		46,		[8,38],			good
[1.01,1.5],		40,		[8,32],			bad
[1.01,1.5],		36,		[8,28],			bad				

[1.1,1.3],		41,		[7,34],			good		yes
[1.1,1.3],		41,		[8,33],			bad		yes
[1.1,1.3],		40,		[8,32],			bad
[1.1,1.3],		36,		[8,28],			bad
[1.1,1.3],		36,		[5,31],			bad

[1.01,1.3],		45,		[8,37],			good
[1.01,1.3],		45,		[5,40],			bad
[1.01,1.3],		40,		[8,32],			bad
[1.01,1.3],		36,		[8,28],			bad
[1.01,1.3],		36,		[5,31],			bad
-----------------------------------------------------------------------------------------------------

2) Variance-based ROIs
2.1) 2 ROIs: 0.75 variance threshold. 23 and 25 locations per ROI

The first reported row for each design threshold corresponds to D-opt mimic Network design algorithm
The last row for each design threshold corresponds to the minimum number of sensors for each ROI
----------------------------------------------------------------------------------------------------
design threshold,	n_sensors,	n_sensors/ ROI,		Dopt-performance	figure
[1.3,1.5],		46,		[21,25],		good			yes	
[1.3,1.5],		46,		[23,23],		bad			yes
[1.3,1.5],		43,		[23,20],		bad			yes
[1.3,1.5],		36,		[23,13],		bad
[1.3,1.5],		36,		[17,19],		bad

[1.1,1.5],		47,		[22,25],		good			yes
[1.1,1.5],		47,		[23,24],		bad			yes
[1.1,1.5],		36,		[23,13],		bad
[1.1,1.5],		36,		[17,19],		bad

[1.01,1.5],		47,		[22,25],		good
[1.01,1.5],		47,		[23,24],		bad
[1.01,1.5],		36,		[23,13],		bad
[1.01,1.5],		36,		[17,19],		bad

------------------------------------------------------------------------------------------------------
method	design threshold	n_sensors	optimal	figure
IRNet	[1.3,1.5]		46		yes	no
Dopt	[1.3,1.5]		46		yes	no

IRNet	[1.5,2.0]		44		yes	yes
Dopt 	[1.5,2.0]		46		yes	yes
Dopt	[1.5,2.0]		44		no	yes

IRNet	[2.0,2.5]		40		yes	yes
Dopt	[2.0,2.5]		45		yes	yes
Dopt	[2.0,2.5]		40		no	yes


------------------------------------------------------------------------------------------------------


2.2) 3 ROIs: 0.5 and 0.75 variance thresholds. 5,18,25 locations per ROI

The first reported row for each
------------------------------------------------------------------------------------------------------
method	design threshold	n_sensors	optimal		figure
IRNet	[1.3,1.5,2.0]		46		yes		yes
Dopt	[1.3,1.5,2.0]		48		yes		yes
Dopt	[1.3,1.5,2.0]		46		no		yes

IRNet	[1.5,2.0,2.5]		43		yes		yes
Dopt 	[1.5,2.0,2.5]		48		yes		yes
Dopt	[1.5,2.0,2.5]		43		no		yes



