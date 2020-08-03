import setuptools
from setuptools import setup


metadata = {'name': 'm_utils',
            'maintainer': 'Edward Aziz',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'A tiny package of some useful tools for modeling-related data processing',
            'license': 'MIT',
            'url': 'https://github.com/redjerdai/m_utils',
            'download_url': 'https://github.com/redjerdai/m_utils',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '0.0.6.0',
            'long_description': '',
            'python_requires': '>=3.6'}


setup(**metadata)
