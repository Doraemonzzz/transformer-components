from setuptools import setup, find_packages

setup(
  name='taco',
  version='0.0.0',
  description='Transformer components',
  author='Doraemonzzz',
  author_email='doraemon_zzz@163.com',
  url='https://github.com/Doraemonzzz/Trev-Xformers',
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
    )
)