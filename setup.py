from setuptools import setup, find_packages

setup(
    name='tacos',
    version='0.0.2',
    description='Transformer components',
    author='Doraemonzzz',
    author_email='doraemon_zzz@163.com',
    url='https://github.com/Doraemonzzz/transformer-components',
    install_requires=[
        'torch',
        'einops',
    ],
    packages=find_packages(
        exclude=[
                "examples",
                "examples.*",
                "data",
                "data.*",
        ]
    ),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

)