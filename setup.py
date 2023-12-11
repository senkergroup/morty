import os
from setuptools import setup, Extension
import numpy
import morty

external_files = []
os.chdir('morty')
for root, dirs, files in os.walk('external'):
    external_files.append(root + '/*')
os.chdir(os.pardir)

setup(
    name='morty',
    version=morty.__version__,
    maintainer='Carsten Tschense',
    maintainer_email='carsten.tschense@uni-bayreuth.de',
    description='Molecular modelling and NMR toolkit for Python',
    #ext_modules=cythonize(Extension('morty.analytical.exsy_csa', sources=['morty/analytical/exsy_csa' + ext], include_dirs=[numpy.get_include()])),
    packages=('morty', 'morty.analytical', 'morty.modeling', 'morty.calculate', 'morty.util'),
    package_data={'morty' : external_files},
    requires=['numpy (>=1.20)', 'scipy (>=1.11)'],
    python_requires='>=3.9'
)
