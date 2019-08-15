
# 1 WSL 安装
1. 控制面板->程序->添加功能->基于Linux的子系统
2. MicroSoft Store->搜索WSL/Linux->选择安装
3. 启动->用户名（只接受小写，可以强行使用大写）及密码
# 2 Linux 配置

0. 更新软件源:
    ```bash
    sudo cp /etc/apt/sources.list /etc/apt/sources.list.backup
    sudo vi /etc/apt/sources.list
    # 更换为清华镜像或者其他国内软件源提升速度
    ...
    source /etc/apt/sources.list
    sudo apt update 
    sudo apt upgrade
    ```
1. 添加C/C++环境:
    ```bash
    sudo apt install GCC
    sudo apt install G++
    sudo apt install GDB
    ```
2. python 环境
    下载anaconda并安装，安装时注意，如果开启了root用户，请退出root用户并且将安装包文件移出root用户目录后再安装，否则如果后期需要开启图形界面并使用图形化anaconda将会报错。去清华镜像更换anaconda软件源
    ```bash
    # 创建新的python3.7虚拟环境
    conda create --name env_name python=python_version 
    # 安装新的库
    conda insatll pakage_name
    # 卸载已安装的库
    conda uninstall pakage_name
    # 列出已安装的库
    conda list
    # 切换虚拟环境
    source activate env_name
    ```
3. 图形界面安装
    1. Windows远程连接+xrdp

        Linux端：

            ```bash
            sudo apt install xfce4
            sudo apt install xrdp
            # 配置xrdp
            sudo sed -i ‘s/port=3389/port=3390/g’ /etc/xrdp/xrdp.ini
            # 配置xsession
            sudo echo xfce4-session >~/.xsession
            # 启动服务
            sudo service xrdp restart
            ```

        Windows端：

        1. 打开远程服务
        2. 计算机输入：`Localhost:3390`/`127.0.0.1:3390`(3390即之前配置的端口号，可以更改)
        3. 用户名： Linux端的用户名，或者是`root`，也可以之后再设置
        4. 点击连接，输入linux的用户名/root以及密码/root密码

    2. VcXsrv

        1. linux 端

            ```bash
            # sudo apt install xorg
            sudo apt install xfce4
            # 设置显示界面
            export DISPLAY=:0.0
            export LIBGL_ALWAYS_INDIRECT=1
            ```
            或者添加到`~/.bashrc`（推荐）

        2. Windows端：
            1. 安装[VcXsrv](https://sourceforge.net/projects/vcxsrv/)
            2. `DISPLAY Number`选择`0`，与之前export的要一致
            3. start no client
            4. 勾选Disable access control
        3. startxfce4

4. 本地主机显示远程GUI界面
    1. 安装[VcXsrv](https://sourceforge.net/projects/vcxsrv/)，并配置，远程连接时输入`export DISPLAY=:0.0`，然后运行程序即可

