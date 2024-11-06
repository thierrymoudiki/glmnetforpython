# -*- coding: utf-8 -*-
"""
def loadGlmLib():
=======================
INPUT ARGUMENTS:

                NONE

=======================
OUTPUT ARGUMENTS: 

glmlib          Returns a glmlib object with methods that are equivalent 
                to the fortran functions in GLMnet.f
=======================
"""
import ctypes
import os
import subprocess

def loadGlmLib():

    # Define paths to the shared library files
    glmnet_so = "GLMnet.so"
    #glmnet_dll = os.path.join(current_directory_path, "GLMnet.dll")

    try:
        if os.name == "posix":  # For Unix-like systems
            print(f"\n\n Loading shared library: {glmnet_so}")
            glmlib = ctypes.cdll.LoadLibrary(glmnet_so)
        elif os.name == "nt":  # For Windows systems
            print(f"Loading shared library: {glmnet_dll}")
            glmlib = ctypes.cdll.LoadLibrary(glmnet_dll)
        else:
            raise OSError("Unsupported operating system")
        print("Library loaded successfully.")
        return glmlib
    except OSError as e:
        print(f"Failed to load library: {e}")
        return None