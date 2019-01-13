<!-- TOC -->

- [远程库设置](#%E8%BF%9C%E7%A8%8B%E5%BA%93%E8%AE%BE%E7%BD%AE)
- [Git 操作](#git-%E6%93%8D%E4%BD%9C)
- [分支](#%E5%88%86%E6%94%AF)

<!-- /TOC -->

# 远程库设置

1. github 创建repo：

2. 为电脑创建 SSH Key:
    ```git
    ssh-keygen -t rsa -C "youremail@example.com"
    ```
3. 设置全局用户名以及邮箱：
    ```git
    git config --global user.name "your name"
    git config --global user.email "youremail@example.com"
    ```
4. clone 远程库：
    ```git
    git clone git@github.com:github_username/repo_name.git
    ```
>注：若是已有本地库，再与远程库链接，则需要先在GitHub 创建新repo，再将本地库与远程库链接：
>>git remote add origin git@github.com:username/repositoryname.git

# Git 操作

* 添加：
    ```git
    git add file/directory
    ```
* 提交
    ```git
    git commit -m "info"
    ```
* 推送
    ```git
    git push
    // git push master origin
    ```
* 查看状态
    ```git
    git status
    ```
* 删除
    ```git
    git rm file/dir
    git commit -m "delete"
    ```
* 撤销修改

    ```git
    git checkout --file
    ```
* 版本回退

    ```git
    git reset HEAD file  //撤销工作区所有修改
    git reset --hard HEAD^  //退回上一个版本，^^回退两个版本，~10 回退10个版本
    ```
# 分支
* 冲突解决，先拉取云端更新，再push:

    ```git 
    git pull 
    git push
    ```
