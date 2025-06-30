# Automatic DMI Experiment Report

Manual extraction and analysis of signals from biomedical imaging data, such as MRI and DMI, is a time-consuming and error-prone process. After each invivo experiment, there is a need to summarize it in a report. Although the raw acquired data is preprocessed in Matlab, in order to create this report there is a need to load the datasets, manually identify anatomical regions, and compute relevant signal metrics — all while juggling multiple tools and scripts. This project aims to streamline and automate that workflow by providing a user-friendly graphical interface for visualizing MRI and DMI data, interactively selecting regions of interest (ROIs), and automatically generating clear, reproducible reports. By reducing manual effort and improving consistency, this tool accelerates data analysis and improves productivity in preclinical and clinical imaging research.

## 🔍 What Does This Project Do?

This project is a Python-based tool designed to streamline the analysis of biomedical imaging data by automating the process of extracting and reporting key information from MRI and DMI scans. It takes a .mat file containing 3D volumetric data over time and collects relevant experiment details from the user. The tool then generates a comprehensive report that includes summarized experiment metadata, overlaid visualizations of selected MRI and DMI slices, and interactive tools for segmenting or selecting specific organs or regions of interest (ROIs). For each selected region, the tool calculates and plots the average signal over time, providing researchers with an efficient, reproducible way to interpret dynamic imaging data.
  
## 📥 Input & Output

- **Input**
  
The input to this tool is a .mat file containing 4D imaging data — typically MRI and DMI volumes over time — along with any relevant experimental metadata provided by the user (e.g., subject ID, scan parameters, time points).
- **Output**
  
The output is a structured PDF report that includes: a summary of the experiment, overlayed MRI and DMI slices for selected anatomical views, interactive segmentation results, and plots of average signal intensities over time for each selected region of interest (ROI).

## 🛠️ Technical Details
Written in Python, using libraries such as h5py or scipy.io for loading .mat files, numpy and scipy for processing, and matplotlib for visualization. An optional GUI is available using PyQt5 or napari for interactive image viewing and region selection.

Runs via command-line interface (CLI) or a simple graphical interface (GUI).

Input files are read locally from .mat files containing MRI and DMI 4D volumes (3D over time). All outputs, including overlay figures, plots, and a final report, are saved to a structured project folder.

Dependencies can be installed via:

pip install -r requirements.txt

For Segmentation:
Install Segment Anything:
 
pip install git+https://github.com/facebookresearch/segment-anything.git


## 📦 How to Download, Install, and Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/mayak1988/Automatic_DMI_Experiment_Report.git
   cd Automatic_DMI_Experiment_Report

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
3. Install the SAM (Segment Anything):
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git

4. Download the sam_checkpoint and place its path in the config\config.py file:
   [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
  
   
6. Run the program with an example command:
   ```bash
   python main.py


This will launch the GUI. You can then load a .mat file, enter experiment details, select MRI/DMI slices, choose regions of interest (ROIs), and generate a full report automatically.



************************************************************************************************************
This project was developed as part of the WIS Python programming course at the Weizmann Institute of Science. 
