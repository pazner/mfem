#!/usr/bin/env bash

# BL3
gmsh -format msh22 -2 BL3.geo

# visc_BGM45-15
gmsh -format msh22 -2 visc_BGM45-15.geo -o visc_BGM45-15.msh
gmsh -format msh22 -order 3 -2 visc_BGM45-15.geo -o visc_BGM45-15_o3.msh

# naca
gmsh -format msh22 -2 naca0012_bl.geo -o naca0012_bl.msh
gmsh -format msh22 -order 3 -2 naca0012_bl.geo -o naca0012_bl_o3.geo
