#!/usr/bin/env python

from distutils.core import setup

setup(
    name='midi-ml',
    version='0.0.1',
    author='Zac Pustejovsky',
    author_email='zaqk.putsky@gmail.com',
    description='Python models and midi data extraction pipelines built for CUNY GC Machine Learning course',
    packages=[
        'midi_ml.models',
        'midi_ml.pipelines',
        'midi_ml.tools',
        'test'
    ]
)
