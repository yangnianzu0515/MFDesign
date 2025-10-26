# Calculate the rmsd.
We use the align.py to cal calculate the RMSD between two structures.
```
python align.py
```
Need to modified the dirpath, because the logical of main function is write for our early file structure. 
If your file structure changed, May need to modified some logcial that read pdb files.

# Calculate the DockQ.
* Note: Group the antibody chains and the antigen chains, repectively. 
Thus May different with the af3 or chai-1.
* Need to modified the specific paths.
* Advice that using the ebm_main function to calculate the DockQ. 
```
python dockq_cal.py
```

# Eval more.
* Need to notice that we need more than one eval metric to evaluate model,
thus first try building a env to eval.
<!-- conda install -c conda-forge openmm
conda install omnia::eigen3 -->

