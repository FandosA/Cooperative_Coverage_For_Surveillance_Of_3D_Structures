# Cooperative Coverage For Surveillance Of 3D Structures

The python file provided implements the algorithm of the paper [Cooperative coverage for surveillance of 3D structures](https://ieeexplore.ieee.org/document/8205999).

It simulates an autonomous multi-robot system to cover and surveil a 3D structure (a cylinder in this case). You only have to download it, and play with the parameters, the rest is already implemented. You can also comment/uncommment the functions at the end of the ```main``` function to plot the velocities, trajectories, etc.

The landmarks are randomly distributed along the surface of the cylinder. Each robot has assigned a set of landmarks, and each landmark has assigned an only robot. The initial positions and orientations of the robots, as well as the initial positions and assignments of the landmarks are shown in the following image:

<img src="https://user-images.githubusercontent.com/71872419/124789999-1141d380-df4b-11eb-8e1c-46dae3c512e9.png" width="300" height="300"> <img src="https://user-images.githubusercontent.com/71872419/124790884-d7250180-df4b-11eb-845c-d85a12d191f9.png" width="300" height="300">


The positions and orientations of the landmarks randomy distributed are shown in the next images:

<img src="https://user-images.githubusercontent.com/71872419/124789146-4d286900-df4a-11eb-90d4-f19a56fbf4e2.png" width="300" height="300"> <img src="https://user-images.githubusercontent.com/71872419/124789154-50235980-df4a-11eb-86a2-43a2fb69bf73.png" width="300" height="300">


Finally, after running the experiment, if the parameters are correctly set, something like this is obtained:

<img src="https://user-images.githubusercontent.com/71872419/124791218-2703c880-df4c-11eb-8e62-bf6f5c82db95.png" width="200" height="200"> <img src="https://user-images.githubusercontent.com/71872419/124791264-3125c700-df4c-11eb-81e7-52d4f59dddca.png" width="170" height="200"> <img src="https://user-images.githubusercontent.com/71872419/124791486-629e9280-df4c-11eb-8b38-4c067a4ad82f.png" width="200" height="200"> <img src="https://user-images.githubusercontent.com/71872419/124791542-6f22eb00-df4c-11eb-949b-c837de35ec99.png" width="170" height="200">


In order to have an idea about how the progress of the velocities and positions of the robots along the iterations, I also plotted them:

<img src="https://user-images.githubusercontent.com/71872419/124792393-43543500-df4d-11eb-9745-2c778bbd0a8f.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792407-464f2580-df4d-11eb-9897-088882f89da6.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792427-494a1600-df4d-11eb-9c42-8a7961519a88.png" width="250" height="250">

<img src="https://user-images.githubusercontent.com/71872419/124792745-94642900-df4d-11eb-80fb-b1c124b2a5e7.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792765-9a5a0a00-df4d-11eb-8cde-5d7118934843.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792770-9c23cd80-df4d-11eb-9646-efe1dadb667d.png" width="250" height="250">

<img src="https://user-images.githubusercontent.com/71872419/124792780-9f1ebe00-df4d-11eb-8ebd-d978ba88ed42.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792788-a04feb00-df4d-11eb-967d-431065eadf7c.png" width="250" height="250"> <img src="https://user-images.githubusercontent.com/71872419/124792806-a47c0880-df4d-11eb-84d3-093ce33b6cca.png" width="250" height="250">


> Adaldo, A. et al. ???Cooperative coverage for surveillance of 3D structures???. In: *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*. 2017, pp. 1838???1845. doi: [10.1109/IROS.2017.8205999](https://ieeexplore.ieee.org/document/8205999)
