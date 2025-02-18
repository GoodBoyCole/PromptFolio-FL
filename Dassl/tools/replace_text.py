"""
Replace text in python files.
"""
import glob
import os.path as osp
import argparse
import fileinput

EXTENSION = ".py"


def is_python_file(filename):
    ext = osp.splitext(filename)[1]
    return ext == EXTENSION


def update_file(filename, text_to_search, replacement_text):
    print("Processing {}".format(filename))
    with fileinput.FileInput(filename, inplace=True, backup="") as