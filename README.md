Title of the project - "B-Cell Epitope prediction using SVM"
----------------------------------------------------------------------------

Brief description of the project
---------------------------------
 
In this project, we developed a machine learning model for linear B-cell epitopes using support vector machine(SVM) classifier algorithm with RBF kernel.
The model was trained utilizing standard datasets containing equal number of epitope and non-epitope sequences and tested using independent test sets.
The performance of the model was evaluated using metrics such as accuracy score.

Datasets
---------

The dataset consists of equal number of epitopes and non-epitope sequences.It is available in the 'Dataset' folder.
 
Language and Tools used for Development
---------------------------------------

Programming language: python3
Python modules and libraries: sklearn, numpy, pandas

Steps to run the programs
--------------------------------

The file named 'b_cell_biprofile.py' contains the code for the prediction model.

Two more files named 'freq.py' and 'biprofile.py' were used as support files for data pre-processing and to generate csv files using pre-processed data for training and testing purposes.

These files can be run using any python IDE such as PyCharm(community edition) by opening the files in the IDE and clicking the run button.

The files can also be run in command prompt using the command : python 'filename.py' provided that python is installed and all the module requirements are fulfilled. 
For example: To run 'b_cell_biprofile.py' in the command prompt type the command : python b_cellbiprofile.py and press enter.

If encountered with an error such as 'ModuleNotFoundError' during the execution of the program. Please install the missing module displayed in the error message using the command: pip install modulename and re-run the program.

For example: If an error ModuleNotFoundError: No module named 'sklearn' is displayed then install the 'sklearn' library by using the command: pip install sklearn in the command prompt and once the installation is complete re-run the program.
 
On succesful execution, the output is displayed on the terminal and it contains a sample of the dataset used and the accuracy score of the model and other important metrics.
