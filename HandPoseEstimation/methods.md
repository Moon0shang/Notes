

# 1.Classification by method

## 1.1 Random Forest

略。。。

## 1.2 2D CNN

## 1.3 3D CNN

# 2.Classification by data type

## 2.1 color image -> 2D joints

## 2.2 color image -> 3D joints

## 2.3 depth image -> 2D joints

## 2.4 depth image -> 3D joints

## 2.5 3D volumetric representation -> 3D joints

## 2.6 point cloud -> 3D joints

### HandPointNet [[PDF]](papers/HandPointNet.pdf) [[Code]](https://sites.google.com/site/geliuhaontu/HandPointNet.zip?attredirects=0&d=1)

overview:
<div align="center">
<img src=./Stuffs/HandPointNet.jpg alt=HandPointNet>
</div>

1. data preprocess

    1. depthmap to point cloud, see reference in [part 3.1](#31-depthmap---point-cloud)
    2. Oriented bounding box(OBB), see [part 3.2](#32-obb)
    3. data normalize,see [part 3.3](#33-normalize)

2. net architecture
    main architecture: `PointNet++`,see [part 4.1]()

# 3. Reference Formula

## 3.1 Depthmap -> Point Cloud

- require data: `depthmap(u,v): (h,w)`, `point cloud(x,y,z): (h*w,3)`
- other parameter: `focal: fx,fy`, `principle(center) point: ux,uy`
- transfer formula (for each pixel/points): 
  
$$\begin{aligned}
x &= (u-ux)/fx \\
y &= (v-uy)/fy \\
z &= depthmap(u,v)
\end{aligned}$$

## 3.2 OBB

- require data: `point cloud p(x,y,z): (h*w,3)`
- formula:

$$\begin{aligned}   
     coeff,score,latent &= pca(p) \\
     R_{obb} &= cross(coeff(:,3),coeff(:,1)) \\
     p_{obb}&={R_{rot}}^T\times p
\end{aligned}$$  
## 3.3 normalize
- require data: `point cloud p(x,y,z): (h*w,3)`
- formula:

$$\begin{aligned}
    mean &= mean(p,axis=1)\\
    l_{max} &= \max(x_{max}-x_{min},y_{max}-y_{min},z_{max}-z_{min})\\
    p_{nor}&=(p-mean)/l_{max}
\end{aligned}$$

# 4 Net Architecture

## 4.1 PointNet++

<div align="center">
<img src=Stuffs/HandPointNet.jpg alt=pointnet++>
</div>
