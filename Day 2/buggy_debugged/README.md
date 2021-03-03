# Project
Optimization of a python code used to analyse magnetic data of .raw files from our MPMS system. Exercises are also present in the subfolders

Using our own lab. we can acquire data from the Magnetic Properties Measurement System of Quantum design.

Usually, every point is directly calculated by fitting the raw data to the one expected from a magnetic dipole. However this approach is not possible for most of my samples since a persistent background appears on top of the sample's signal. I already possess some code that enables me to do basic treatment such as those but I would like to :

- Make them clearer (for instance by putting all of the basic functions in separated modules to be imported from)

- Use classes (e.g : M.magnetization(), M.field()) since this approach makes more sense in this case

- Try to do the same code but by implementing interpolation from SciPy modules.

- Enable it to be used by entering only (1) the .raw and .diag (diagnostic) files from the MPMS and (2) the suspected position of the sample.
