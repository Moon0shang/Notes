<span id="top"></span>
<!-- TOC -->

- [标题](#标题)
- [文字样式](#文字样式)
- [输入公式](#输入公式)
- [插入图片/链接](#插入图片链接)
- [注脚&缩写](#注脚缩写)
- [列表](#列表)
- [代码](#代码)
- [表格](#表格)

<!-- /TOC -->

# 标题

```markdown
# 标题
## 二级标题
### 三级标题
#### 四级标题
##### 五级标题
###### 六级标题
```

# 文字样式

- 注释
    ```markdown
    <!---
    注释
    注释
    --->
    ```
- 首行缩进：
    ```markdown
    1. 半方大的空白:&ensp;或&#8194;
    2. 全方大的空白:&emsp;或&#8195;
    3. 不断行的空白格:&nbsp;或&#160;
    ```

    首行缩进1&ensp;首行缩进2&#8194;首行缩进3&emsp;首行缩进4&#8195;首行缩进5&#160;首行缩进6&nbsp;首行缩进

    &emsp;&emsp;首行缩进
- 文本居中
    <center>居中</center>

    >使用这种方法居中的文本如果包含其他markdown
- 特殊文本
    ```markdown
    *斜体*
    **粗体**
    ==高亮==
    ~~删除线~~
    H_2_O 下标
    2^10^ 上标
    > 引用
    ```
- 字体样式
    ><font 更改语法>   你的内容   </font>
    >color=#0099ff   更改字体颜色
    >face="黑体"   更改字体
    >size= 7     更改字体大小

    <font face="黑体">我是黑体字</font>
    <font face="微软雅黑">我是微软雅黑</font>
    <font face="STCAIYUN">我是华文彩云</font>
    <font color=#0099ff size=7 face="黑体">color=#0099ff size=72 face="黑体"</font>
    <font color=#00ffff size=72>color=#00ffff</font>
    <font color=gray size=72>color=gray</font>
    <font color=red size=9> SCP </font>

# 输入公式

在markdown中输入公式采用Latex编译，因此语法与Latex基本一致。围绕公式插`$`，单个表示行内公式，两个表示行间公式，以进入公式编辑环境：

```markdown
$\alpha$
$$\alpha$$
```

>$\alpha$
>$$\alpha$$

# 插入图片/链接

在markdown中插入的图片/链接可以是网络内容上的链接，也可以是本文中内容的链接

- A [link(baidu)](https://www.baidu.com) 文字链接
- 页内链接跳转[标题](#标题)
    >1. []中括号填写需要在页面上显示的内容； 
    >2. ()小括号内部声明跳转目标标题，以#开头，标题题号如果包含`.`、下划线直接忽略掉，标题文本中如果有空格，使用`-`横杠符号替代，标题文本中的大写字母转换成小写。
- 转至页面[顶部](#top)
    ```markdown
    在文档顶部设置锚点：<span id="jump"></span>
    ```
- 插入图片(有些语法不适用所用编辑器，此处只有适用于VS Code 的语法)
    ![a](../loss_increasing.png)
- 插入图片并设置图片大小
    <img src=../loss_increasing.png width = "400" height = "260" alt=ll>
- 插入图片并居中，同时可以设置图片大小，添加标题
    <div align="center">
    <img src=../loss_increasing.png alt=ll>

    标题
    </div>

# 注脚&缩写

- Some text with a footnote.[^1]
- Markdown converts text to HTML.

*[HTML]: HyperText Markup Language
[^1]:the footnote.

# 列表

1. 无序列表
    - item
        * item
            + item  
2. 有序列表
    1. item
    2. item
    3. item
3. 待办列表
    - [ ] incomplete item
    - [x] complete item

# 代码

- 单行代码/行内代码some `inline code`
- 代码块：
    ```python
    # a code block with highlight
    a = int('10')
    ```
- 如果行首缩进超过当前等级4个空格，就会被识别为代码段

        like this

some

    something like this

# 表格

- 方法一
    ```markdown
    | item      | value |
    | --------- | ----- |
    | item1     | v1    |
    | item22222 | v2    |
    ```
- 方法二
    | column1 | column2 |
    | ------- | ------- |
    | 111     | 2222    |
- 对齐，居中`:-:`，向左对齐`:--`，向右对齐`--:`
    | 不对齐 | 居中对齐 | 向左对齐 | 向右对齐 |
    | ------ | :------: | :------- | -------: |
    | abc    |   abc    | abc      |      abc |

# 分隔符

```markdown 
-------
```
