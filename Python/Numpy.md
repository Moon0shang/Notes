
# 技巧&常用

1. 矩阵的自我复制
   * np.repeat
    ```python {cmd}
    import numpy as np

    a = np.array([[0, 1], [2, 3]])
    b = np.repeat(a,2,axis=0)
    print('b:\n',b)
    c = np.repeat(a,2,axis=1)
    print('c:\n',c)
    ```
   * np.tile
    ```python {cmd}
    import numpy as np

    a = np.array([[0, 1], [2, 3]])
    b = np.tile(a,[2,1])
    print('b:\n',b)
    ```