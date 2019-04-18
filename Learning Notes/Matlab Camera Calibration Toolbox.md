
# 安装及注意事项

## 安装和说明文档
**获取途径**： 说明文档[Documentation](http://www.vision.caltech.edu/bouguetj/calib_doc/)以及下载地址[Download](http://www.vision.caltech.edu/bouguetj/calib_doc/download/index.html)。下载压缩包解压出来即可使用，建议解压到`Matlab`安装目录中的`toolbox`文件夹中，然后在`Matlab`的`设置路径`中添加整个`toolbox`到路径。


## 注意事项
1. 使用时输入`pwd`获取当前工作文件夹或者打开文件夹作为当前工作文件夹；
2. 在命令行输入`calib_gui`或`calib`来启动GML标定工具，正式开始使用前将需要标定的图片移动到工作文件夹内，*不能放在文件夹里面*
3. 用于标定的图片至少为10张比较好，使用大于等于A3纸张打印棋盘图片
![棋盘格](./stuffs/棋盘格.jpg)

# 标定

## 单目标定
1. 将图片放到当前工作目录之后，命令行输入`calib_gui`打开标定软件，选择标准模式`standard`，一般不会出现内存不够的情况；
![模式选择](stuffs/calib1.jpg)
2. 点击`imagename`查看当前目录中的图片，
![查看图片](stuffs/calib2-1.jpg)

导入时需要输入文件名前面不包含数字的部分，
![命令行显示](stuffs/calib2-2.jpg)

此时matlab会弹出你选择的图片的组合界面，位置是根据输入图片的数字编号来决定的，并不会影响结果；
![paras](stuffs/calib2-3.jpg)
3. 单击`Extract grid corners`，
![Extract](stuffs/calib3-1.jpg)

参数选择默认即可，
![paras](stuffs/calib3-2.jpg)

然后根据提示用鼠标将所有图片的棋盘的角点点出来，
![ori](stuffs/calib3-3.jpg)![click](stuffs/calib3-4.jpg)![finish](stuffs/calib3-5.jpg)

选择完后，第一张会询问是否更改其他参数，默然即可，
![comm](stuffs/calib3-7.jpg)

然后会出来两张提取角点后的图片，
![c1](stuffs/calib3-6.jpg)![c2](stuffs/calib3-8.jpg)

对所有图片进行同样的操作；
4. 单击`calibration`，
![calib](stuffs/calib4-1.jpg)

命令行会输出计算的相机的内参；
![intrinsic](stuffs/calib4-2.jpg)
5. 单击`Comp. Extrinsic`，
![comp](stuffs/calib5-1.jpg)

此时会要求输入要计算外参的图片的全名(不包含扩展名)以及图片格式，
![input](stuffs/calib5-2-1.jpg)

其他参数默认即可，
![paras](stuffs/calib5-2-2.jpg)

接下来按之前的方法框出四个角点，
![box](stuffs/calib5-3.jpg)![all](stuffs/calib5-4.jpg)

选择完后会输出两张图片，
![p1](stuffs/calib5-5.jpg)![p2](stuffs/calib5-6.jpg)

命令行会输出外参参数值，
![extrinsic](stuffs/calib5-7.jpg)

至此获取相机内外参数步骤结束。

