# Cooperative Coverage For Surveillance Of 3D Structures

The python file provided implements the algorithm of the paper [Cooperative coverage for surveillance of 3D structures](https://ieeexplore.ieee.org/document/8205999).

It simulates an autonomous multi-robot system to cover and surveil a 3D structure (a cylinder in this case). You only have to download it, and play with the parameters, the rest is already implemented.

The landmarks are randomly distributed along the surface of the cylinder. Each robot has assigned a set of landmarks, and each landmark has assigned an only robot. The initial positions and orientations of the robots, as well as the initial positions and assignments of the landmarks are shown in the following image:

![InitialPos_Agents_front](https://user-images.githubusercontent.com/71872419/124787818-1736b500-df49-11eb-9af6-cba869280527.png)

The orientations of the landmarks are shown in the next images:

<img src="https://user-images.githubusercontent.com/71872419/124789146-4d286900-df4a-11eb-90d4-f19a56fbf4e2.png" width="100" height="100">
<img src="https://user-images.githubusercontent.com/71872419/124789154-50235980-df4a-11eb-86a2-43a2fb69bf73.png" width="100" height="100">

![FrontView_Landmarks](https://user-images.githubusercontent.com/71872419/124789146-4d286900-df4a-11eb-90d4-f19a56fbf4e2.png){:height="50%" width="50%"}
![TopView_Landmarks](https://user-images.githubusercontent.com/71872419/124789154-50235980-df4a-11eb-86a2-43a2fb69bf73.png){:height="75%" width="75%"}

