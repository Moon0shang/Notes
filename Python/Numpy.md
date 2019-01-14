
# 技巧&常用

1. 矩阵的自我复制
   * np.repeat
    ```python
    import numpy as np

    a = np.array([[0, 1], [2, 3]])
    b = np.repeat(a,2,axis=0)
    # out: [[0 1]
    #       [0 1]
    #       [2 3]
    #       [2 3]]
    c = np.repeat(a,2,axis=1)
    # out: [[0 0 1 1]
    #       [2 2 3 3]]
    ```
   * np.tile
    ```python
    import numpy as np

    a = np.array([[0, 1], [2, 3]])
    b = np.tile(a,[2,1])
    #out [[0 1]
    #     [2 3]
    #     [0 1]
    #     [2 3]]
    ```