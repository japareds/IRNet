figures folder contains 2 sub-folders according to the method used for defining the Regions of Interest (ROI)
- Folder distance based applies a distance threshold from Ciutadella station. Set to 10km
- Folder variance_based applies a variance threshold to Fully-monitored coordiante error variance. Set to 0.75 if 2 ROIs and 0.5,0.75 if 3 ROIs

---------------------------------------------------------
The results of the Network design algorithm correspond to a certain number of sensors (n_ND)
The results of the Dopt algorithm correspond to a certain number of sensors (n_Dopt)
Usually, n_ND < n_Dopt
The first figure shows the coordinate error variance for both results.
The second figure shows the results for both results but Dopt set to n_ND as well. The distribution of sensors changes.
