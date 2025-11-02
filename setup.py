"""
Setup script for the Plagiarism Checker package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="plagiarism-checker",
    version="1.0.0",
    author="Sukhraj Singh",
    description="A Python tool to detect plagiarism using TF-IDF and cosine similarity",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Sukhraj-Singh-2006/Plagiarism-Checker-",
    py_modules=["plagiarism_checker", "cli"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Text Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "plagiarism-checker=cli:main",
        ],
    },
)
