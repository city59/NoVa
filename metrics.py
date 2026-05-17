from collections import defaultdict  # 导入默认字典，用于处理键不存在的情况
import heapq  # 导入堆队列模块，用于实现优先队列
import math  # 导入数学函数
from tqdm import tqdm  # 导入进度条模块
from numba import jit  # 导入即时编译装饰器，用于加速计算


class Metric(object):
    """推荐系统评估指标类，包含各种评估推荐系统性能的指标计算方法"""

    def __init__(self):
        pass  # 初始化方法，当前不需要初始化任何属性

    @staticmethod
    def hits(origin, res):
        """计算每个用户的命中数
        
        Args:
            origin: 原始数据，用户真实交互的物品集合
            res: 推荐结果，为每个用户推荐的物品列表
            
        Returns:
            hit_count: 每个用户的命中数量字典
        """
        hit_count = {}  # 存储每个用户的命中数
        for user in origin:
            items = list(origin[user].keys())  # 用户实际交互的物品列表
            predicted = [item[0] for item in res[user]]  # 为用户推荐的物品列表
            hit_count[user] = len(set(items).intersection(set(predicted)))  # 计算交集大小，即命中数
        return hit_count

    @staticmethod
    def hit_ratio(origin, hits):
        """
        计算总体命中率
        
        Note: 这种类型的命中率计算公式为：
         (测试集中检索到的交互数 / 测试集中所有交互数)
        
        Args:
            origin: 原始数据，用户真实交互的物品集合
            hits: 每个用户的命中数字典
            
        Returns:
            命中率，范围[0,1]
        """
        total_num = 0  # 测试集中的总交互数
        for user in origin:
            items = list(origin[user].keys())  # 用户实际交互的物品列表
            total_num += len(items)  # 累加所有用户的交互数
        hit_num = 0  # 命中的交互总数
        for user in hits:
            hit_num += hits[user]  # 累加所有用户的命中数
        return hit_num / total_num  # 返回命中率

    @staticmethod
    def precision(hits, N):
        """计算精确率(Precision)，即推荐的物品中有多少比例是用户感兴趣的
        
        Args:
            hits: 每个用户的命中数字典
            N: 推荐列表长度
            
        Returns:
            精确率，范围[0,1]
        """
        prec = sum([hits[user] for user in hits])  # 所有用户的命中总数
        return prec / (len(hits) * N)  # 总命中数除以(用户数*推荐列表长度)

    @staticmethod
    def recall(hits, origin):
        """计算召回率(Recall)，即用户感兴趣的物品有多少比例被推荐
        
        Args:
            hits: 每个用户的命中数字典
            origin: 原始数据，用户真实交互的物品集合
            
        Returns:
            召回率，范围[0,1]
        """
        recall_list = [hits[user] / len(origin[user]) for user in hits]  # 每个用户的召回率
        recall = sum(recall_list) / len(recall_list)  # 所有用户召回率的平均值
        return recall

    @staticmethod
    def F1(prec, recall):
        """计算F1值，精确率和召回率的调和平均数
        
        Args:
            prec: 精确率
            recall: 召回率
            
        Returns:
            F1值，范围[0,1]
        """
        if (prec + recall) != 0:
            return 2 * prec * recall / (prec + recall)  # 精确率和召回率的调和平均数
        else:
            return 0  # 避免除零错误

    @staticmethod
    def MAE(res):
        """计算平均绝对误差(Mean Absolute Error)
        
        Args:
            res: 评分预测结果列表，每个元素包含预测评分和真实评分
            
        Returns:
            MAE值，越小越好
        """
        error = 0  # 累计误差
        count = 0  # 评分数量
        for entry in res:
            error += abs(entry[2] - entry[3])  # 累加绝对误差，entry[2]为真实评分，entry[3]为预测评分
            count += 1
        if count == 0:
            return error  # 避免除零错误
        return error / count  # 返回平均绝对误差

    @staticmethod
    def RMSE(res):
        """计算均方根误差(Root Mean Square Error)
        
        Args:
            res: 评分预测结果列表，每个元素包含预测评分和真实评分
            
        Returns:
            RMSE值，越小越好
        """
        error = 0  # 累计平方误差
        count = 0  # 评分数量
        for entry in res:
            error += (entry[2] - entry[3])**2  # 累加平方误差，entry[2]为真实评分，entry[3]为预测评分
            count += 1
        if count == 0:
            return error  # 避免除零错误
        return math.sqrt(error / count)  # 返回均方根误差

    @staticmethod
    def NDCG(origin, res, N):
        """计算归一化折损累积增益(Normalized Discounted Cumulative Gain)
        NDCG考虑了推荐物品在列表中的位置，位置越靠前的物品权重越大
        
        Args:
            origin: 原始数据，用户真实交互的物品集合
            res: 推荐结果，为每个用户推荐的物品列表
            N: 推荐列表长度
            
        Returns:
            NDCG值，范围[0,1]，越大越好
        """
        sum_NDCG = 0  # 所有用户NDCG值的总和
        for user in res:
            DCG = 0  # 折损累积增益
            IDCG = 0  # 理想情况下的折损累积增益
            # 1 = 相关, 0 = 不相关
            for n, item in enumerate(res[user]):
                if item[0] in origin[user]:  # 如果推荐的物品在用户的交互列表中
                    DCG += 1.0 / math.log(n + 2, 2)  # 根据位置计算权重
            for n, item in enumerate(list(origin[user].keys())[:N]):
                IDCG += 1.0 / math.log(n + 2, 2)  # 计算理想情况下的DCG
            sum_NDCG += DCG / IDCG  # 归一化，计算单个用户的NDCG
        return sum_NDCG / len(res)  # 返回所有用户NDCG的平均值


def ranking_evaluation(origin, rec_list, N):
    """评估排序推荐结果的性能
    
    Args:
        origin: 原始数据，用户真实交互的物品集合
        rec_list: 推荐结果，为每个用户推荐的物品列表
        N: 推荐列表长度
        
    Returns:
        包含多个评估指标的字典
    """
    predicted = {}  # 截取推荐列表前N个物品
    for user in rec_list:
        predicted[user] = rec_list[user][:N]
    if len(origin) != len(predicted):
        print('测试集和预测集的长度不匹配！')
        exit(-1)  # 如果用户数不匹配，退出程序
    if len(origin) == 0:
        return {  # 如果原始数据为空，返回默认值
            'Hit_Ratio': -1,
            'Precision': -1,
            'Recall': -1,
            'NDCG': -1,
        }

    measure = {}  # 存储各项评估指标
    hits = Metric.hits(origin, predicted)  # 计算命中数
    hr = Metric.hit_ratio(origin, hits)  # 计算命中率
    prec = Metric.precision(hits, N)  # 计算精确率
    recall = Metric.recall(hits, origin)  # 计算召回率
    NDCG = Metric.NDCG(origin, predicted, N)  # 计算NDCG
    measure['Hit_Ratio'] = hr
    measure['Precision'] = prec
    measure['Recall'] = recall
    measure['NDCG'] = NDCG
    return measure


def rating_evaluation(res):
    """评估评分预测结果的性能
    
    Args:
        res: 评分预测结果列表
        
    Returns:
        包含MAE和RMSE指标的列表
    """
    measure = []  # 存储评估结果
    mae = Metric.MAE(res)  # 计算平均绝对误差
    measure.append('MAE:' + str(mae) + '\n')  # 添加MAE结果
    rmse = Metric.RMSE(res)  # 计算均方根误差
    measure.append('RMSE:' + str(rmse) + '\n')  # 添加RMSE结果
    return measure


@jit(nopython=True)  # 使用numba加速计算
def find_k_largest(K, candidates, id2item=None):
    """查找数组中最大的K个元素
    
    Args:
        K: 需要查找的元素数量
        candidates: 候选分数列表
        id2item: 候选物品的ID列表，默认为None
        
    Returns:
        ids: 最大K个元素的索引列表
        k_largest_scores: 最大K个元素的分数列表
    """
    # 找前 k 大，稳定排序（优先 score 最大，其次 item id 最大）
    n_candidates = []  # 存储候选项
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))  # 添加前K个元素

    heapq.heapify(n_candidates)  # 将列表转换为最小堆

    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:  # 如果当前分数大于堆顶元素
            # 比堆顶元素大，入堆
            heapq.heapreplace(n_candidates, (score, iid + K))  # 替换堆顶元素
    n_candidates.sort(key=lambda d: d, reverse=True)  # 按分数降序排序
    ids = [item[1] for item in n_candidates]  # 提取索引
    k_largest_scores = [item[0] for item in n_candidates]  # 提取分数
    return ids, k_largest_scores


def cal_metrics(c_data, top_k=100):
    """计算推荐系统的评估指标
    
    Args:
        c_data: 包含用户、物品、标签和分数的DataFrame
        top_k: 推荐列表长度，默认为100
        
    Returns:
        包含多个评估指标的字典
    """
    # 构造 ground truth（真实数据）
    origin = defaultdict(dict)  # 使用默认字典存储用户-物品交互
    for user_id, item_id in c_data.query('label == 1')[['user_id',
                                                        'item_id']].values:
        origin[user_id][item_id] = 1  # 标记用户与物品的正向交互

    # print('---> only pos user: ',
    #       len(c_data.query(f'user_id in {list(origin.keys())}')),
    #       ', all data: ', len(c_data))
    # 产出预测分数
    rec_list = {}  # 存储每个用户的推荐列表
    for user_id, group in tqdm(c_data.query(
            f'user_id in {list(origin.keys())}').groupby('user_id'),
                               disable=True):  # 按用户分组，并禁用进度条
        candicates = group['score'].tolist()  # 候选物品的分数列表
        id2item = group['item_id'].tolist()  # 候选物品的ID列表

        item_ids, scores = find_k_largest(top_k, candicates, id2item)  # 查找分数最高的top_k个物品
        rec_list[user_id] = list(zip(item_ids, scores))  # 将物品ID和分数组合为推荐列表

    return ranking_evaluation(origin, rec_list, top_k)  # 评估推荐结果
