# Neural Semantic Surface Maps (NSSM)

[Paper](https://arxiv.org/pdf/2309.04836.pdf)
&nbsp;&nbsp;
[Project Page](https://geometry.cs.ucl.ac.uk/projects/2024/nssm/)
&nbsp;&nbsp;
[Fast forward](https://youtu.be/y7bPZz_5bfw)

---

This repository contains code to train a Neural Semantic Surface Maps and a [Neural Surface Maps](https://geometry.cs.ucl.ac.uk/projects/2021/neuralmaps/).


# Setup

To set up the environment you can download the docker image and build it or install all the required packages in your conda environment (see docker image).

```bash
docker build --build-arg username=luca --build-arg userid=`id -u` -t nssm ./nssm/
```
This will create a docker image, the sudo password is `docker`.


Once you have the docker image up and running, run the docker container and link it to your workspace to access the code.
```
docker run --security-opt seccomp=unconfined -h DOCKER --name nssm  -v ~/workspace:/home/luca/workspace --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -it --shm-size=2gb nssm:latest bash
```
Please adjust the path to your workspace!

NOTE: the instructions above create a user called `luca` and mount the workspace in the user folder, if you specify a different username then change it also when running the docker container.

This configuration was tested on Ubuntu 20.04 with a 2080Ti.


Now compile the C++ code to parametrize a mesh:
```bash
cd parametrization_cpp
mkdir build
cd build
cmake .. 
make -j4
```


# Preprocessing

Create a folder for the shape pair, inside it create a folder called `meshes` containing the source and target mesh (both obj files).

```bash
cd workspace
mkdir nssm_pair
mkdir meshes
cp source.obj nssm_output/meshes/source.obj
cp target.obj nssm_output/meshes/target.obj
```

That's it, you are ready to optimize a map between them!

NOTE: the code will look for folder called `meshes` containing a `source.obj` and `target.obj` file. If this does not happen, then the whole pipeline will crash.

# Optimization

You can run the full pipeline with just one script:

```bash
cd nssm
./run_pair.sh ~/workspace/nssm_pair
```

Remember, the path must contain a `meshes` folder as described in the preprocessing step.
The code will automatically generate and save inside the folder `nssm_pair` the output data for each stage.
The map is contained inside the `nssm_pair/map` folder (both model weights and meshes).

---

# Bibtex

```
@article{morreale2024neural,
    title={Neural Semantic Surface Maps},
    author={Morreale, Luca and Aigerman, Noam and Kim, Vladimir G. and Mitra, Niloy J.},
    booktitle={Computer Graphics Forum},
    volume={43},
    number={2},
    year={2024},
    organization={Wiley Online Library}
}
```
