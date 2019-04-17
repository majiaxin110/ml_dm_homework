# 运行环境
Python3
所依赖的包：
scipy
scikit-learn
matplotlib
numpy
IPython
若要执行ass_B/history内程序，依赖的包还有tensorflow和PIL。ass_B/history内是对数据集3进行分类和尝试聚类的探索过程

# 运行方式
## 任务一
ass_A文件夹内AssA.py定义了封装Sci2014算法的函数
FSFDP(agg_data: np.ndarray, distance_func, dc=None, t=0.02, gama_graph: bool = False, isSave: bool=False, cluster_cores: np.ndarray = None)
**agg_data**:要聚类的数据
**distance_func**:距离计算函数
**dc**:截断距离
**t**:截断距离计算参数，t与dc应只传入一个
**gama_graph**:是否输出γ图形
**isSave**:是否将聚类结果保存至result/task1.csv
**cluster_cores**:聚类中心数组，若该参数非None，函数将返回全部点的聚类结果

需要使用该算法的程序导入此文件后即可使用
ass_A文件夹内main.py定义了对../data/Aggregation.txt文件的读写和聚类操作，可在Python Console下使用如下语句

``` 
runfile('ass_A/main.py', wdir='ass_A')
```
## 任务二
类似地，需将原始数据文件置于data文件夹内，先执行ass_B/preProcess.py对数据作预处理。随后执行ass_B/main.py，该文件内程序将调用各种聚类算法并输出结果图形，聚类结果数组也将被保存至相应位置。
执行ass_B/final_data_collector.py将读取main.py的结果数据，输出各个聚类算法的轮廓系数，并将结果数据组合，将全部类别输出至assB.csv

