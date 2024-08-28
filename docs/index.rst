.. notebook_test documentation master file, created by
   sphinx-quickstart on Sat Jul 25 11:56:56 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Research Computing and Software Developpment
===========================================

| *Author*: Boris Bolliet

.. note::
   Contains Teaching Material for MPhil DiS 2024/2025 at the University of Cambridge 


Schedule 
----------------------------------------

+------------------------------------------+---------------------------------------------------------------+
| **Date**                                 | **Notebook**                                                  |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, X October 2024, 15:00-16:00   | Tutorial x: Introduction to X                           |
+------------------------------------------+---------------------------------------------------------------+
| Tuesday, X. November 2024, 15:00-16:00   | Tutorial x: Introduction to Y                              |
+------------------------------------------+---------------------------------------------------------------+

How to run the notebooks
------------------------

- **Locally**: To run the notebooks locally, follow these steps:
  1. Install Anaconda or Miniconda on your system if you haven't already.
  2. Create a new conda environment using the provided YAML file:
     ```
     conda env create -f C1_cpu.yml  # For CPU-only systems
     # OR
     conda env create -f C1_gpu.yml  # For systems with NVIDIA GPUs
     ```
  3. Activate the environment:
     ```
     conda activate C1
     ```
  4. Navigate to the directory containing the notebooks.
  5. Launch Jupyter Lab:
     ```
     jupyter lab
     ```
  6. In the Jupyter Lab interface, open the desired notebook.

  Alternatively, if you prefer using Jupyter Notebook instead of Jupyter Lab, replace step 5 with:
     ```
     jupyter notebook
     ```

- **Google Colab**: from a web browser, `Google Colab <https://colab.research.google.com/notebooks/intro.ipynb#recent=true>`_. 

Lectures
--------------------------

.. toctree::
   :caption: Lectures
   :maxdepth: 2

   material/lecture1/MakingNeuralNetsEmulators
   material/lecture2/MakingNeuralNetsEmulators

.. toctree::
   :caption: Miscellaneous
   :maxdepth: 2

   material/misc/MakingNeuralNetsEmulators
