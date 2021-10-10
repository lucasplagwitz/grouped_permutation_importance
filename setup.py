try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages


__version__ = '0.0.1'

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='grouped_permutation_importance',
    packages=find_packages(),
    include_package_data=True,
    version=__version__,
    description="Permutation Importances for Feature Groups.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Lucas Plagwitz',
    author_email='lucas.plagwitz@uni-muenster.de',
    download_url="https://github.com/lucasplagwitz/grouped_permutation_importance/archive/' + __version__ + '.tar.gz",
    url= "https://github.com/lucasplagwitz/grouped_permutation_importance.git",
    keywords=['machine learning', 'feature importance', 'permutation importance', 'feature groups'],
    install_requires=[
        'numpy',
        'scikit-learn',
        ]
)