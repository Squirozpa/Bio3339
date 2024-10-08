from setuptools import setup, find_packages

setup(
    name='Bio3339_tools',
    version='0.1.0',
    packages=find_packages(include=['Bio3339_tools', 'Bio3339_tools.*']),
    install_requires=[
        'numba',
        'numpy'],
    python_requires='>=3.10',
    description='A collection of tools for Bio3339',
    author='Sebastián Quiroz',
    author_email='squirozpa@uc.cl',
    url='https://github.com/squirozpa/Bio3339',
    classifiers=[
        'Programming Language :: Python :: 3.12',
        'License :: Dunno yet',
        'Operating System :: OS Independent',
    ],
)
