

# Argparse


- 作用：用于解析命令行

- 用法：

    ```python
    # args.py
    import argparse

    def cmd():
        args = argparse.ArgumentParser(
            description='Personal Information ', epilog='Information end ')
        # 必写属性,第一位
        args.add_argument("name", type=str, help="Your name")
        # 必写属性,第二位
        args.add_argument("birth", type=str, help="birthday")
        # 可选属性,默认为None
        args.add_argument("-r", '--race', type=str, dest="race", help=u"民族")
        # 可选属性,默认为0,范围必须在0~150
        args.add_argument("-a", "--age", type=int, dest="age",
                        help="Your age", default=0, choices=range(150))
        # 可选属性,默认为male
        args.add_argument('-g', "--gender",   type=str, dest="gender",
                        help='Your gender',         default='male', choices=['male', 'female'])
        # 可选属性,默认为None,-p后可接多个参数
        args.add_argument("-p", "--parent", type=str, dest='parent',
                        help="Your parent",      default="None", nargs='*')
        # 可选属性,默认为None,-o后可接多个参数
        args.add_argument("-o", "--other", type=str, dest='other',
                        help="other Information", required=False, nargs='*')

        args = args.parse_args()  # 返回一个命名空间,如果想要使用变量,可用args.attr
        print("argparse.args=", args, type(args))
        print('name = %s' % args.name)
        d = args.__dict__
        for key, value in d.items():
            print('%s = %s' % (key, value))


    if __name__ == "__main__":
        cmd()
    ```
- 使用：
    ```cmd
    python args.py -h   
    python args.py xiaoming 1991.11.11
    python args.py xiaoming 1991.11.11 -p xiaohong xiaohei -a 25 -r han -g female -o 1 2 3 4 5 6
    ```
- `add_argment`参数：
    - dest：如果提供dest，例如`dest=”a”`，那么可以通过`args.a`访问该参数
    - nargs：参数的数量，*(任意多个，可以为0个)，+(一个或更多)，值为?时，首先从命令行获得参数，如果有-y后面没加参数，则从const中取值，如果没有-y，则从default中取。
        ```python
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('x', nargs='?',default='default')
        >>> parser.add_argument('-y', nargs='?',const='const', default='default')
        >>> parser.parse_args('1 -y 2'.split())
        Namespace(x='1', y='2')
        >>> parser.parse_args('1 -y'.split())
        Namespace(x='1', y='const')
        >>> parser.parse_args([])
        Namespace(x='default', y='default')
        ```
    - default：设置参数的默认值
    - const：保存一个常量
    - type：把从命令行输入的结果转成设置的类型
    - choice：允许的参数值
        ```python
        parser.add_argument(“-v”, “–verbosity”, type=int,
        choices=[0, 1, 2], help=”increase output verbosity”)
        ```
    - help：参数命令的介绍
    - action：参数触发的动作
    - store：保存参数，默认
    - store_const：保存一个被定义为参数规格一部分的值（常量），而不是一个来自参数解析而来的值。
    - store_ture/store_false：保存相应的布尔值
    - append：将值保存在一个列表中。
    - append_const：将一个定义在参数规格中的值（常量）保存在一个列表中。
    - count：参数出现的次数
        ```python
        parser.add_argument(“-v”, “–verbosity”, action=”count”,
        default=0, help=”increase output verbosity”)
        ```
    - version：打印程序版本信息
- 注意：
    >参数的访问形式为`args.name`，如果`dest`存在，则优先通过`dest`设置来访问参数，再通过其他的来访问参数；对于`-n`，`--name`如果同时存在，则只能使用`--name`来访问参数。
