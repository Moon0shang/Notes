
参考[CSDN-aipiano](https://blog.csdn.net/aichipmunk/article/details/9264703)

# 摄像头的标定

1. 参考[Calibration](./Matlab_Camera_Calibration_Toolbox.md)

    按照标定的步骤分别获取深度图（若是使用`libfreeenect`来获取的数据流，此处使用**IR帧**来代替深度图片，因为深度图片中无法分辨棋盘格的格子）以及彩色图的内参（焦距(focal center)$fc_x,fc_y$，主要点(camera center/principle point)$cc_x,cc_y$）以及外参（旋转矩阵rotate matrix，转换矩阵transfer matrix）


2. 由上一步获取的内外参数，构成内参矩阵以及外参矩阵
    **内参矩阵**：$intrinsic\ matrix:H_{ir}=
    \left[
        \begin{matrix}
            fc_x&0&cc_x\\
            0&fc_y&cc_y\\
            0&0&1
        \end{matrix}
    \right]$

    **外参矩阵**：旋转矩阵 $rotate\ matrix: R_{img} = Rc\_ext$；
            转换矩阵 $rotate\ matrix: T_{img}=Tc\_ext$

# 配准

1. 设$P_{ir}$为在深度摄像头坐标下某点的空间坐标，$p_{ir}$为该点在像平面上的投影坐标（$x, y$单位为像素，$z$等于深度值，单位为毫米），$H_{ir}$为深度摄像头的内参矩阵，由小孔成像模型可知，他们满足以下关系:
   $$p_{ir}=H_{ir}P_{ir}\\
   P_{ir}=H^{-1}_{ir}p_{ir}$$
又设$P_{rgb}$为在RGB摄像头坐标下同一点的空间坐标，$p_{rgb}$为该点在RGB像平面上的投影坐标，$H_{rgb}$为RGB摄像头的内参矩阵。由于深度摄像头的坐标和RGB摄像头的坐标不同，他们之间可以用一个旋转平移变换联系起来，即：
   $$P_{rgb} = RP_{ir}+T$$
其中$R$为旋转矩阵，$T$为平移向量。最后再用H_rgb对P_rgb投影，即可得到该点对应的RGB坐标：
   $$p_{rgb}=H_{rgb}P_{rgb}$$
需要注意的是，p_ir和p_rgb使用的都是齐次坐标，因此在构造p_ir时，应将原始的像素坐标（x，y）乘以深度值，而最终的RGB像素坐标必须将p_rgb除以z分量，即（x/z，y/z），且z分量的值即为该点到RGB摄像头的距离（单位为毫米）。
2. 外参矩阵实际上也是由一个旋转矩阵R_ir（R_rgb）和平移向量T_ir（T_rgb）构成的，它表示将一个全局坐标系下的点P变换到摄像头坐标系下，分别对深度摄像头和RGB摄像头进行变换，有以下关系：
$$\begin{array}{l}
    P_{rgb} = RP_{rgb}+T_{rgb}\\
    P_{ir} = RP_{ir}+T_{ir}
\end{array}$$
在第一式中，将P用P_ir、R_ir和T_ir表示，并带入第二式，可得：
    $$P_{rgb}=R_{rgb}R^{-1}_{ir}+T_{ragb}-R_{rgb}R^{-1}_{ir}T_{ir}$$
从上式可以看出，这是在将P_ir变换为P_rgb，对比之前的式子：
    $$P_{rgb} = RP_{ir}+T$$
可得：
    $$\begin{array}{l}
        R = R_{rgb}R^{-1}_{ir}\\
        T = T_{rgb}-R_{rgb}R^{-1}_{ir}T_{ir}
    \end{array}$$
因此，我们只需在同一场景下，得到棋盘相对于深度摄像头和RGB摄像头的外参矩阵，即可算出联系两摄像头坐标系的变换矩阵（注意，所有旋转矩阵都是正交阵，因此可用转置运算代替求逆运算）。虽然不同场景下得到的外参矩阵都不同，计算得到的$R$和$T$也有一些变化，但根据实际实验结果来看，使用一个正面棋盘的标定图像就可达到较好的效果
