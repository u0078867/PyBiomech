Requirements:
-------------

A minimal Python environment with *setuptools* library is expected. We recommend *Anaconda*.
*PyBiomech* was tested on:

- Anaconda Python 2.7 64-bit
- VTK 5.10.1
- NumPy 1.12.0
- SciPy 0.18.1

Installation (for Anaconda)
---------------------------

* Open a command shell
* Type the following to install the first dependencies:
  ```shell
  conda install numpy scipy vtk
  ```
  Then choose "y".
* Install the last dependency like this:
  ```shell
  easy_install btk
  ```
* Proceed with the installation of *PyBiomech*:
  ```shell
  pip install PyBiomech --no-deps
  ```
* To verify than everything went ok, type:
  ```shell
  python
  ```
  and then import the library:
  ```py
  >>> import PyBiomech
  ```
  If no error occurs, then you have correctly installed it!

Update
------

* Open a command shell
* Type the following to update *PyBiomech*:
  ```shell
  pip install PyBiomech --no-deps
  ```
