from setuptools import setup, find_packages

setup(name='backbone', 
      version='0.1',
      packages=find_packages(),
      url='https://github.com/yizaochen/backbone_rigidity.git',
      author='Yizao Chen',
      author_email='yizaochen@gmail.com',
      license='MIT',
      install_requires=[
          'jupyterlab',
          'pandas',
          'numpy',
          'matplotlib'
      ]
      )
