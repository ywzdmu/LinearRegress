using System;
using System.Collections.Generic;
using System.Data;
using System.Text;
using System.Linq;

namespace LinearRegress_ywzdmu
{
    /// <summary>
    /// 线性回归&梯度下降
    /// ywz 2019-07-12 17:47:23
    /// </summary>
    public class LinearRegressWithBGD
    {
        //学习率和最大迭代次数
        double _learnningRate;
        int _maxInterations;

        //训练数据（训练集+真实值,真实值在最后）和列说明
        DataTable _trainData;
        Mask[] _trainMask;

        //训练得到的参数
        double[] thetas;

        //训练的评价指标
        double _score;
        public double Score
        {
            get { return _score; }
            set { _score = value; }
        }

        //待预测数据和列说明
        DataTable _predictData;
        Mask[] _predictMask;



        /// <summary>
        /// 构造
        /// </summary>
        /// <param name="trainData"></param>
        /// <param name="trainMask"></param>
        /// <param name="predictData"></param>
        /// <param name="predictMask"></param>
        /// <param name="learningRate"></param>
        /// <param name="maxInterations"></param>
        public LinearRegressWithBGD(DataTable trainData, Mask[] trainMask, DataTable predictData, Mask[] predictMask, double learningRate = 0.001, int maxInterations = 100000)
        {
            _score = -1;
            _trainData = trainData;
            _trainMask = trainMask;
            _predictData = predictData;
            _predictMask = predictMask;
            _learnningRate = learningRate;
            _maxInterations = maxInterations;
        }


        /// <summary>
        /// 训练
        /// </summary>
        /// <returns></returns>
        public string Fit()
        {
            string message = string.Empty;

            try
            {
                //基本校验 待完善
                if (_trainData == null || _trainMask == null || _predictData == null || _predictMask == null ||
                    _trainData.Rows.Count < 1 || _trainData.Columns.Count < 1 || _predictData.Rows.Count < 1 || _predictData.Columns.Count < 1)
                {
                    return string.Format("对象初始化数据存在空值");
                }
                if (_trainData.Columns.Count != _trainMask.Length)
                {
                    return string.Format("训练集数据列数[{0}]与列属性说明数量[{1}]不符", _trainData.Columns.Count, _trainMask.Length);
                }
                if (_predictData.Columns.Count != _predictMask.Length)
                {
                    return string.Format("待预测数据列数[{0}]与列属性说明数量[{1}]不符", _predictData.Columns.Count, _predictMask.Length);
                }

                //训练集onehot
                double[,] _xTrainTransform;
                double[] _yTrainTransform;
                DataTransform(_trainData, _trainMask, out _xTrainTransform, out _yTrainTransform);

                if (_xTrainTransform == null || _yTrainTransform == null)
                {
                    return string.Format("特征处理时出错");
                }

                //训练集数据标准化
                double[,] _xTrainNormalization;
                DataNormalization(_xTrainTransform, out _xTrainNormalization);

                if (_xTrainNormalization == null)
                {
                    return string.Format("数据标准化时出错");
                }

                //开始梯度下降求解
                thetas = LinearRegressionWithBGD(_xTrainNormalization, _yTrainTransform);

                if (thetas == null)
                {
                    return string.Format("线性回归算法未能得到正确的thetas参数值");
                }

                //计算评价分数
                List<double> trainPredictScore = new List<double>();
                for (int i = 0; i < _xTrainNormalization.GetLength(0); i++)
                {
                    double trainPredict = Hypothesis(thetas, _xTrainNormalization, i);
                    trainPredictScore.Add(1 - Math.Abs(trainPredict - _yTrainTransform[i]) / _yTrainTransform[i]);
                }

                _score = trainPredictScore.Average();
            }
            catch (Exception ex)
            {
                message = string.Format("训练时发生异常_信息[{0}]", ex.Message);
            }

            return message;
        }

        /// <summary>
        /// 预测
        /// </summary>
        /// <param name="predictResult"></param>
        /// <returns></returns>
        public string Predict(out Dictionary<string, double> predictResult)
        {
            predictResult = null;
            string message = string.Empty;

            try
            {
                //基本校验 待完善
                if (_score < 0)
                {
                    return string.Format("请先调用Fit方法训练模型");
                }

                if (thetas == null)
                {
                    return string.Format("线性回归算法未能得到正确的thetas参数值");
                }

                //待预测数据集稀疏化
                double[,] _xPredictTransform;
                double[] _xPredictKey;
                DataTransform(_predictData, _predictMask, out _xPredictTransform, out _xPredictKey);

                if (_xPredictTransform == null || _xPredictKey == null || _xPredictTransform.GetLength(0) != _xPredictKey.Length)
                {
                    return "处理待预测数据失败";
                }

                if (_xPredictTransform.GetLength(1) + 1 != thetas.Length)
                {
                    return "待预测数据稀疏化后参数个数与thetas个数不同";
                }

                //待预测数据集标准化
                double[,] _xPredictNormalization;
                DataNormalization(_xPredictTransform, out _xPredictNormalization);

                if (_xPredictNormalization == null)
                {
                    return string.Format("数据标准化时出错");
                }

                //计算预测值
                predictResult = new Dictionary<string, double>();
                for (int i = 0; i < _xPredictNormalization.GetLength(0); i++)
                {
                    predictResult.Add(_xPredictKey[i].ToString(), Math.Round(Hypothesis(thetas, _xPredictNormalization, i), 2));
                }
            }
            catch (Exception ex)
            {
                predictResult = null;
                message = string.Format("训练时发生异常_信息[{0}]", ex.Message);
            }

            return message;
        }


        /// <summary>
        /// 进行线性回归计算_BGD求解
        /// </summary>
        /// <param name="xData"></param>
        /// <param name="yData"></param>
        private double[] LinearRegressionWithBGD(double[,] xData, double[] yData)
        {
            //样本数量
            int sampleNum = xData.GetLength(0);
            //参数数量
            int parameterNum = xData.GetLength(1) + 1;

            double[] thetas = new double[parameterNum];
            double[] thetas_updated = new double[parameterNum];
            thetas.SetValue(0, 0);
            thetas_updated.SetValue(0, 0);

            //循环做梯度下降
            int inter = 0;
            bool goon = true;
            while (inter < _maxInterations && goon)
            {
                goon = false;

                //求解每个thetas
                for (int j = 0; j < parameterNum; j++)
                {
                    //遍历所有样本求解CostFunction对于各个参数的偏微分
                    double differentialThetas = 0.0;
                    for (int m = 0; m < sampleNum; m++)
                    {
                        if (j == 0)
                        {
                            differentialThetas = differentialThetas + Hypothesis(thetas, GetSingleLine(xData, m)) - yData[m];
                        }
                        else
                        {
                            differentialThetas = differentialThetas + (Hypothesis(thetas, GetSingleLine(xData, m)) - yData[m]) * xData[m, j - 1];
                        }
                    }

                    //做梯度下降计算
                    thetas_updated[j] = thetas[j] - _learnningRate * differentialThetas / sampleNum;
                }

                //查看step大小
                for (int i = 0; i < parameterNum; i++)
                {
                    if (Math.Round(thetas[i], 5) != Math.Round(thetas_updated[i], 5))
                    {
                        goon = true;
                        break;
                    }
                }

                //更新当前thetas
                thetas_updated.CopyTo(thetas, 0);

                inter++;
            }

            return thetas;
        }


        /// <summary>
        /// 计算预测值
        /// </summary>
        /// <param name="thetas"></param>
        /// <param name="xData"></param>
        /// <returns></returns>
        private double Hypothesis(double[] thetas, double[] xData)
        {
            double result = thetas[0];

            for (int i = 1; i < thetas.Length; i++)
            {
                result = result + thetas[i] * xData[i - 1];
            }

            return result;
        }
        private double Hypothesis(double[] thetas, double[,] xData, int rowNum)
        {
            double[] tempArray = new double[xData.GetLength(1)];
            for (int i = 0; i < xData.GetLength(1); i++)
            {
                tempArray[i] = xData[rowNum, i];
            }
            return Hypothesis(thetas, tempArray);
        }

        /// <summary>
        /// 获取二维数组的某一行
        /// </summary>
        /// <param name="inputData"></param>
        /// <param name="rowNoum"></param>
        /// <returns></returns>
        private double[] GetSingleLine(double[,] inputData, int rowNoum)
        {
            int length = inputData.GetLength(1);
            double[] result = new double[length];
            for (int j = 0; j < length; j++)
            {
                result[j] = inputData[rowNoum, j];
            }
            return result;
        }


        /// <summary>
        /// 数据稀疏化表示 
        /// 各个巷道堆垛机调整的速度加速度不同 LANEWAY不能直接以数值的形式参与计算
        /// </summary>
        /// <param name="inputData"></param>
        /// <param name="inputMask"></param>
        /// <param name="X"></param>
        /// <param name="Y"></param>
        private void DataTransform(DataTable inputData, Mask[] inputMask, out double[,] X, out double[] Y)
        {
            X = null;
            Y = null;

            int outputRowCount = inputData.Rows.Count;
            int outputColumnCount = 0;

            //统计被标记为字典的列 的数据包含不同值的多少
            Dictionary<int, List<string>> dictDataMark = new Dictionary<int, List<string>>();
            for (int i = 0; i < inputMask.Count(); i++)
            {
                if (inputMask[i] == Mask.X_DICT)
                {
                    List<string> valueList = new List<string>();
                    for (int j = 0; j < inputData.Rows.Count; j++)
                    {
                        string value = inputData.Rows[j][i].ToString();

                        if (!valueList.Contains(value))
                        {
                            valueList.Add(value);
                        }
                    }
                    dictDataMark.Add(i, valueList);
                    outputColumnCount = outputColumnCount + valueList.Count;
                }
                else if (inputMask[i] == Mask.X_DIRECT)
                {
                    outputColumnCount++;
                }
            }

            X = new double[outputRowCount, outputColumnCount];
            Y = new double[outputRowCount];

            //遍历输入数据 根据列说明来选择将字典数据向量化 或者直接复制数据
            int offset = 0;
            for (int j = 0; j < inputData.Columns.Count; j++)
            {
                for (int i = 0; i < outputRowCount; i++)
                {
                    switch (inputMask[j])
                    {
                        case Mask.X_DICT:
                            //将样本中出现的列对应的位置标记为1 比如LANEWAY=2时 将第2列标记为1 达到向量化表示的目的 
                            X[i, offset + dictDataMark[j].IndexOf(inputData.Rows[i][j].ToString())] = 1.0;
                            break;

                        case Mask.X_DIRECT:
                            //简单的无变化直接复制
                            X[i, offset] = double.Parse(inputData.Rows[i][j].ToString());
                            break;

                        case Mask.YorKEY_DIRECT:
                            //Y值 直接复制
                            Y[i] = double.Parse(inputData.Rows[i][j].ToString());
                            break;

                        default: break;
                    }
                }

                //计算偏移量 应对多个DICT形式的样本
                offset = offset + (dictDataMark.Keys.Contains(j) ? dictDataMark[j].Count : 1);
            }
        }

        /// <summary>
        /// 数据标准化 
        /// 0均值1方差[z=(x-mu)/sigma]
        /// </summary>
        /// <param name="inputData"></param>
        /// <param name="dataScaled"></param>
        private void DataNormalization(double[,] inputData, out double[,] dataScaled)
        {
            dataScaled = null;

            int rowCount = inputData.GetLength(0);
            int columnCount = inputData.GetLength(1);

            //均值和方差
            double[] mu = new double[columnCount];
            double[] sigma = new double[columnCount];

            //一次遍历求出列的均值[mu=sum/n]和方差[sigma^2=E(x-mu)^2=Ex^2-(mu)^2]
            for (int j = 0; j < columnCount; j++)
            {
                double tempSum = 0.0;
                double tempSumx2 = 0.0;
                for (int i = 0; i < rowCount; i++)
                {
                    tempSum = tempSum + inputData[i, j];
                    tempSumx2 = tempSumx2 + inputData[i, j] * inputData[i, j];
                }
                mu[j] = (double)tempSum / rowCount;
                sigma[j] = (double)tempSumx2 / rowCount - mu[j] * mu[j];
            }

            //数据标准化
            dataScaled = new double[rowCount, columnCount];
            for (int i = 0; i < rowCount; i++)
            {
                for (int j = 0; j < columnCount; j++)
                {
                    dataScaled[i, j] = (inputData[i, j] - mu[j]) / Math.Sqrt(sigma[j]);
                }
            }
        }


        /// <summary>
        /// 列说明 
        /// </summary>
        public enum Mask
        {
            //字典(LANEWAY)
            X_DICT,
            //直接输出
            X_DIRECT,
            //训练集真实值或者待预测集的KEY字段 直接输出
            YorKEY_DIRECT,
        }
    }

}
