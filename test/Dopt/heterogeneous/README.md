D-optimalty sensor placement results for network splitted into different Regions of Interest (ROIs)
The ROIs are defined by the Network design algorithm. D-opt is just for comparison.
D-opt requires knowing the number of sensors deployed in each ROI according to the Network Design algorithm.

Parameters:
N=48,s=36
-----------------------------------------------------------------------------------------------------
1) distance-based ROIs: BCN and rural
1.1) 2 ROIs: BCN and rural. 10km threshold from Ciutadella station. 8 and 40 locations per ROI

The first reported row for each design threshold corresponds to D-opt mimic Network design algorithm.
The last row for each design threshold corresponds to the minimum number of sensors possible.

1) Distance-based ROIs
1.1) 2ROIs: 0,10km threshold. 8 and  40 elements per ROI
method	design threshold	p	optimal	figure
IRNet	[1.1,1.5]		40	yes	
Dopt	[1.1,1.3]

IRNet	[1.1,1.3]		41	yes
Dopt	[1.1,1.3]

IRNet	[1.5,2.0]		39	yes
Dopt	[1.5,2.0]		38	yes

IRNet	[1.5,3.0]		39	yes
Dopt 	[1.5,3.0]		37	yes

IRNet	[2.0,3.0]		38	yes
Dopt	[2.0,3.0]		37	yes

IRNet	[2.0,5.0]		38	yes
Dopt	[2.0,5.0]		37	yes

1.2) 3 ROIs: 0,5,10 thresholds. 3,5,40 elementss per ROI 
method	design threshold	p	optimal	figure
IRNet	[1.5,3.0,5.0]		38	yes
Dopt	[1.5,3.0,5.0]		36	yes

IRNet	[2.0,3.0,5.0]		38	yes
Dopt	[2.0,3.0,5.0]		36	yes

IRNet	[5.0,3.0,2.0]		39	yes
Dopt	[5.0,3.0,2.0]		38	yes


-----------------------------------------------------------------------------------------------------
2) Variance-based ROIs
2.1) 2 ROIs: 0.75 variance threshold. 23 and 25 locations per ROI

The first reported row for each design threshold corresponds to D-opt mimic Network design algorithm
The last row for each design threshold corresponds to the minimum number of sensors for each ROI

method	design threshold	n_sensors	optimal	figure
IRNet	[1.3,1.5]		46		yes	no
Dopt	[1.3,1.5]		46		yes	no

IRNet	[1.5,2.0]		44		yes	yes
Dopt 	[1.5,2.0]		46		yes	yes
Dopt	[1.5,2.0]		44		no	yes

IRNet	[2.0,2.5]		40		yes	yes
Dopt	[2.0,2.5]		45		yes	yes
Dopt	[2.0,2.5]		40		no	yes




2.2) 3 ROIs: 0.5 and 0.75 variance thresholds. 5,18,25 locations per ROI

method	design threshold	n_sensors	optimal		figure
IRNet	[1.3,1.5,2.0]		46		yes		yes
Dopt	[1.3,1.5,2.0]		48		yes		yes
Dopt	[1.3,1.5,2.0]		46		no		yes

IRNet	[1.5,2.0,2.5]		43		yes		yes
Dopt 	[1.5,2.0,2.5]		48		yes		yes
Dopt	[1.5,2.0,2.5]		43		no		yes

------------------------------------------------------------------------------------------------------
3) Random ROIs
3.1) 2ROIs: seed 0. 24 and 24 locations per ROI

method	design threshold	n_sensors	optimal		figure
IRNet	[1.5,2.0]		38		yes		yes
Dopt	[1.5,2.0]		41		yes		yes
Dopt	[1.5,2.0]		38		no		yes

IRNet	[2.0,3.0]		37		yes		yes
Dopt	[2.0,3.0]		40		yes		yes
Dopt	[2.0,3.0]		37		no		yes

IRNet	[3.0,5.0]		38		yes
Dopt	[3.0,5.0]		40		yes

IRNet	[2.0,5.0]		38		yes		
Dopt	[2.0,5.0]		40		yes

IRNet	[2.0,4.0]		38		yes
Dopt	[2.0,4.0]		40		yes


3.2) 3ROIs. seed 0. 16,16,16 elements per ROI
method	design threshold	n_sensors	optimal		figure
IRNet	[1.5,3.0,5.0]		39		yes		yes
Dopt 	[1.5,3.0,5.0]		40		yes		yes
Dopt	[1.5,3.0,5.0]		39		no		yes

IRNet	[2.0,3.0,5.0]		38		yes
Dopt	[2.0,3.0,5.0]		39		yes
------------------------------------------------------------------------------------------------------



