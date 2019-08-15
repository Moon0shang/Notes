

- `torch_geometric.data->Data(x,edge_index,edge_attr,y,pos,norm,face)` [详细介绍](https://rusty1s.github.io/pytorch_geometric/build/html/modules/data.html)
  - `edge_index (LongTensor, optional)` – Graph connectivity in COO format with shape [2, num_edges]. (default: None) 表示的是所有点之间的连接关系（有连接的点），`edge_index[0]` 和 `edge_index[1]` 代表有连接的点的索引，两个索引一一对应，该对应关系为单向的，如果某两点双向连接，则需要将两点索引在两个维度中都一一对应，如下面的程序以及图片所示：
    ```python
    edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    # Data(x=[3, 1], edge_index=[2, 4])
    ```
    <div align="center">
    <img src='./stuffs/graph.svg' alt='grapg'>
    </div>
    如果已经拥有所有点之间的两点间的一一对应关系矩阵，则需要对此矩阵进行转置并`contiguous`，如下所示：
    
    ```python
    edge_index = torch.tensor([[1,0],
                               [0,1],
                               [1,2],
                               [2,1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    # Data(x=[3, 1], edge_index=[2, 4])
    ```