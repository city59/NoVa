import torch


class GraphSparseMatInterface(object):

    @staticmethod
    def convert_sparse_mat_to_tensor(X):
        """
        将scipy稀疏矩阵转换为torch稀疏张量。

        参数:
            X: scipy稀疏矩阵，表示图的邻接矩阵。

        返回值:
            转换后的torch稀疏张量。
        """
        # 将输入的scipy稀疏矩阵转换为COO格式
        coo = X.tocoo()
        # 创建一个包含行索引和列索引的二维张量
        i = torch.LongTensor([coo.row, coo.col])
        # 将非零元素的数据转换为浮点型张量
        v = torch.from_numpy(coo.data).float()
        # 构造并返回torch稀疏张量，并调用coalesce()确保索引唯一且值正确
        return torch.sparse.FloatTensor(i, v, coo.shape).coalesce()

    @staticmethod
    def normalize_graph_mat(adj_mat: torch.sparse.FloatTensor,
                            add_virtual_node_num: int = 0,
                            return_node_degree: bool = False,
                            norm: str = 'both'):
        """
        对原始图的邻接矩阵进行归一化处理，生成归一化的邻接矩阵。

        参数:
            adj_mat: 稀疏张量，表示图的邻接矩阵。
            add_virtual_node_num: 添加的虚拟节点数量，默认为0。
            return_node_degree: 如果设置为True，则返回节点度数，默认为False。
            norm: 使用'both'或'left'进行归一化处理，默认为'both'。

        返回值:
            归一化后的邻接矩阵（如果return_node_degree为True，则同时返回节点度数）。
        """
        # 获取邻接矩阵的形状
        shape = adj_mat.shape
        # 如果邻接矩阵中没有非零元素，则初始化rowsum为全零张量，并加上虚拟节点的数量
        if adj_mat._nnz() == 0:
            rowsum = torch.zeros(adj_mat.shape[0],
                                 device=adj_mat.device) + add_virtual_node_num
        else:
            # 计算每一列的元素和（即节点的入度），并加上虚拟节点的数量
            rowsum = torch.sparse.sum(adj_mat,
                                      dim=0).to_dense() + add_virtual_node_num

        # 根据不同的归一化方式构建归一化的邻接矩阵
        if norm == 'both' and shape[0] == shape[1]:
            # 使用'both'归一化：计算每个节点的逆平方根度数（1 / sqrt(d1 * d2)）
            d_inv = rowsum ** -0.5
            # 将无穷大的值设为0，避免除以0的情况
            d_inv[torch.isinf(d_inv)] = 0.

            # 构造逆平方根度数的稀疏对角矩阵
            d_mat_inv = torch.sparse.FloatTensor(
                torch.arange(len(d_inv), device=adj_mat.device).repeat(2, 1),
                d_inv)
            # 进行两次稀疏矩阵乘法，得到归一化后的邻接矩阵
            norm_adj_tmp = torch.sparse.mm(d_mat_inv, adj_mat)
            norm_adj_mat = torch.sparse.mm(norm_adj_tmp, d_mat_inv)
        elif norm == 'left':
            # 使用'left'归一化：计算每个节点的逆度数（1 / d1）
            d_inv = rowsum ** -1.0
            # 将无穷大的值设为0，避免除以0的情况
            d_inv[torch.isinf(d_inv)] = 0.

            # 构造逆度数的稀疏对角矩阵
            d_mat_inv = torch.sparse.FloatTensor(
                torch.arange(len(d_inv), device=adj_mat.device).repeat(2, 1),
                d_inv)
            # 进行一次稀疏矩阵乘法，得到归一化后的邻接矩阵
            norm_adj_mat = torch.sparse.mm(d_mat_inv, adj_mat)
        else:
            # 如果归一化方式不在'both'或'left'中，则抛出异常
            raise 'norm need in (both, left)'

        # 调用coalesce()确保归一化后的邻接矩阵索引唯一且值正确
        output = norm_adj_mat.coalesce()
        # 如果需要返回节点度数，则将节点度数与归一化后的邻接矩阵一起返回
        if return_node_degree:
            output = output, rowsum
        return output
