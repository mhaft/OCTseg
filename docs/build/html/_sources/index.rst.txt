.. OCTseg documentation master file, created by
   sphinx-quickstart on Thu Aug  1 18:10:49 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to OCTseg's documentation!
==================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

==================
Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

====================
Training and Testing
====================
.. autofunction:: train.main

========
Boundary
========
loss
----
.. automodule:: boundary.loss
	:members:

ops
---
.. automodule:: boundary.ops
	:members:
	
unet
----
.. automodule:: boundary.boundary
	:members:
	
=======
Utility
=======
confusion matrix
----------------
.. automodule:: util.confusion_matrix
	:members:

load batch
----------
.. automodule:: util.load_batch
	:members:

load data
---------
.. automodule:: util.load_data
	:members:
	
plot log file
-------------
.. automodule:: util.plot_log_file

polar to cartesian
------------------
.. automodule:: util.polar2cartesian
	:members:

process oct folder
------------------
.. automodule:: util.process_oct_folder
	:members:
	
read oct roi file 
-----------------
.. automodule:: util.read_oct_roi_file 
	:members:
