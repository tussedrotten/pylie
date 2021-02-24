# pylie
A small Lie library for Python.

This library represents rotations and poses in 3D.
It also implements group and Lie operations, as well as their Jacobians.

## Install library
You can install this library to your Python environment directly from this repository using `pip`:
```bash
pip install https://github.com/tussedrotten/pylie/archive/master.zip
```

## Dependencies
The library depends upon numpy, which is installed together with pylie when using the command above.

The examples and tests have additional dependencies, see [requirements.txt](requirements.txt).
You can install these dependencies with the following command:
```bash
pip install -r requirements.txt
```

## Examples
- [left_right_perturbations.py](examples/left_right_perturbations.py)
- [mean_pose.py](examples/mean_pose.py)
- [pose_interpolation.py](examples/pose_interpolation.py)
- [poses_and_cameras.py](examples/poses_and_cameras.py)
- [vis_perturbations.py](examples/vis_perturbations.py)
