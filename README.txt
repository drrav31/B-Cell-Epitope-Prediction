----------------------------------------------------------------------------
Title of the project - "A Computational model for Adaptive Immune System"
----------------------------------------------------------------------------

---------------------------------
Brief description of the project
---------------------------------
 
In this project, we developed a machine learning model for linear B-cell epitopes using SVM classifier algorithm.
The model was trained utilizing standard datasets containing equal number of epitope and non-epitope sequences and tested using independent test sets.
The performance of the model was evaluated using metrics such as accuracy score.

---------
Datasets
---------

In this project, we used two datasets consisting of equal number of epitopes and non-epitope sequences.
The first dataset is located in the "First_Dataset" folder and the second dataset is located in the "Second_Dataset" folder.
 
--------------
Requirements
--------------

The project was developed using the python programming language on a machine running windows operating system.

-----------------------
Hardware requirements:
-----------------------

Processor: Intel core i3 or higher
RAM: 4GB atleast or more

-----------------------
Software requirements:
-----------------------

Operating system : Windows 7 or higher
Programming language: python 3(version 3.5 or higher)
Python modules and libraries: sklearn, numpy, pandas, scipy and split

--------------------------------
Steps to run the programs
--------------------------------

The code files(python files) are located in the 'b_cells' directory. 
In the 'b_cells' directory, there are two files named 'b_cell.py' and 'b_cell_biprofile.py' which contain the code for the prediction models.
'b_cell.py' utilizes the dataset from the 'First_Dataset' folder and 'b_cell_biprofle.py' utilizes the dataset from the 'Second_Dataset' folder to train and test the models.
These files can be run using any python IDE such as PyCharm(community edition) by opening the files in the IDE and clicking the run button.
The files can also be run in command prompt using the command : python 'filename.py' provided that python is installed and all the module requirements are fulfilled. 
For example: To run 'b_cell.py' in the command prompt type the command : python b_cell.py and press enter.
If encountered with an error such as 'ModuleNotFoundError' during the execution of the program. Please install the missing module displayed in the error message using the command: pip install modulename and re-run the program.
For example: If an error ModuleNotFoundError: No module named 'sklearn' is displayed then install the 'sklearn' library by using the command: pip install sklearn in the command prompt and once the installation is complete re-run the program.
 
On succesful execution, the output displayed contains a sample of the dataset used and the accuracy score of the model.

----------------------------
Steps to run the .exe files
----------------------------

The 'exe files' folder contains two folders named 'b_cell' and 'b_cell_biprofile'
The 'b_cell' folder contains the 'b_cell.exe' application and the 'b_cell_biprofile' folder contains the 'b_cell_biprofile.exe' application along with all the requirements needed to run the applications.
To run these applications:
 Open command prompt 
 Navigate to the appropriate directories in which these files are located.
 Use the command: 'filename'.exe and press enter to run the required application.
 For example: To run 'b_cell.exe' execute the command: b_cell.exe in the command prompt.

On successful execution, a sample of the dataset used and the accuracy score of the model are displayed.

  











