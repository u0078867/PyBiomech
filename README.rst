Installation for Anaconda
-------------------------

* Open a command shell
* Type the following to install the first dependencies:

  .. code-block:: sh

     conda install numpy scipy vtk

  Then choose "y".
* Install the last dependency like this:

  .. code-block:: sh

     easy_install btk

* Proceed with the installation of *PyBiomech*:

  .. code-block:: sh

     pip install PyBiomech --no-deps

* To verify than everything went ok, type:

  .. code-block:: sh

     python

  and then import the library:

  .. code-block:: python

     >>> import PyBiomech

  If no error occurs, then you have correctly installed it!
