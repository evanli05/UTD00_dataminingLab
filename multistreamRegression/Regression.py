import csv
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
import math
import time
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=None,
                            level=logging.INFO)
logger = logging.getLogger('Baseline')
class Regression(object):
    def __init__(self):
        super(Regression, self).__init__()

    @staticmethod
    def get_dist_base(y_col):
        """
        calculate mu_0 and sigma_0_square
        :param y_col: a vector that contains target values
        :return: mean-base - mu_0, var_base - sigma_0_square
        """
        y_min = min(y_col).tolist()[0][0]
        y_max = max(y_col).tolist()[0][0]
        mean_base = (y_min + y_max) / 2.0
        var_base = ((y_max - mean_base) / 2.0) ** 2.0
        return mean_base, var_base

    @staticmethod
    def read_csv(path, size=None, delimiter=",", header=None):
        data = None
        with open(path) as csvfile:
            count = 0
            if not header:
                reader = csv.reader(csvfile, delimiter=delimiter)
            else:
                reader = csv.DictReader(csvfile, fieldnames=header, 
                                        delimiter=delimiter)
            for row in reader:
                #print row
                tmp = [float(x) for x in row]
                if data is None:
                    data = np.array(tmp)
                else:
                    data = np.vstack((data, tmp))
                count += 1
                if size and count > size:
                    break
            data = np.matrix(data, dtype=np.float64)
            return data

    @staticmethod
    def get_covariance(data_matrix):
        """
        calculate covariance between attributes
        :param data_matrix: M * N matrix where M - number of instances, N- 
        number of features
        :return: covariance - covariance between features. x_mean - an 
        array contains mean value of each attribute
        """
        covariance = np.cov(data_matrix.transpose())
        x_mean = np.mean(data_matrix, 0)
        return covariance, x_mean

    @staticmethod
    def cal_weight(data_matrix):
        """
        weight here is the probability density function p.d.f value for each
        data instance.
        :param data_matrix: M * N matrix where M - number of instances, 
        N- number of features
        :return: a list of weight for each instance in the data matrix
        """
        weight = []
        m, n = data_matrix.shape
        covariance, mean = Regression.get_covariance(data_matrix)
        for i in range(m):
            p = multivariate_normal.pdf(data_matrix[i, :], mean.tolist()[0], 
                                        covariance, allow_singular=True)
            weight.append(p)
        return weight

    @staticmethod
    def get_empirical_mean(data_matrix):
        """
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number
        of features, 
        the last column is target value
        :return: empirical_mean, a one-row matrix of values corresponds to [Y, 
        x1,x2,x3,...,xn, 1]
        """
        m, n = data_matrix.shape
        # initialize parameters
        # m0 = [Y, x1,x2,...,xn, 1]
        m0 = np.zeros(1 + n)
        for i in range(m):
            data_sample = data_matrix[i, :]
            m0[0] += data_sample[0, -1]*data_sample[0, -1]
            m0[-1] += data_sample[0, -1]
            m0[1:-1] += (data_sample[0, -1]*data_sample[:, 0:-1]).tolist()[0]
        m0 /= float(m)
        return np.matrix(m0, dtype=np.float64)

    @staticmethod
    def get_est_mean(m, data_matrix):
        """
        :param m: a row matrix contains current parameter values corresponding to [Y, x1,x2,...,xn, 1]
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :return: mu - MLE estimate value of y for each data point.
        """
        matrix = data_matrix[:, :-1]
        ycol = data_matrix[:, -1]
        m_, n_ = matrix.shape
        weight = Regression.cal_weight(matrix)
        m_yy = m[0, 0]
        m_yx1 = m[0, 1:]
        mu = []
        app_est_mean = np.matrix(np.zeros(n_+2), dtype=np.float64)
        mean_base, var_base = Regression.get_dist_base(ycol)
        for i in range(m_):
            sample = matrix[i, :]
            w = weight[i]
            if w == float("inf"):
                if m_yy == 0:
                    m_yy = 1e-10
                tmp = sample.tolist()[0]
                tmp.append(1)
                mean_y = -m_yx1*np.matrix(tmp, dtype=np.float64).transpose()/m_yy
                mean_y = mean_y.tolist()[0][0]
                var_y = 0
            else:
                tmp = sample.tolist()[0]
                tmp.append(1)
                mean_y = 1 / (2 * w * m_yy + (1 / var_base)) * (-2 * w * m_yx1 
                             * np.matrix(tmp, dtype=np.float64).transpose() + 
                             (1 / var_base) * mean_base)
                var_y = 1 / (2 * w * m_yy + (1 / var_base))
                mean_y = mean_y.tolist()[0][0]
            mu.append(mean_y)
            add_var_y = np.matrix(np.zeros(n_+2), dtype=np.float64)
            add_var_y[0, 0] = var_y
            app_est_mean += np.matrix([mean_y] + sample.tolist()[0] + [1], 
                                      dtype=np.float64) * mean_y + add_var_y
        app_est_mean /= float(m_)
        return app_est_mean, mu

    @staticmethod
    def get_update_value_and_error(app_est_mean, empirical_mean, m, lambdas):
        """  
        :param app_est_mean: matrix of app_est_mean
        :param m: a row matrix contains current parameter values corresponding to [Y, x1,x2,...,xn, 1]
        :param empirical_mean: matrix of empirical_mean
        :param lambdas
        :return: app_gradient: a row matrix representing gradient
        """
        avg_error = 0
        app_gradient = app_est_mean - empirical_mean - 2 * lambdas * m
        avg_error += sum(abs(app_gradient).tolist()[0])
        avg_error /= float(app_gradient.shape[1])
        return app_gradient, avg_error

    @staticmethod
    def update_m(m, step, decay, speed, data_matrix, estimated_mean=None, 
                 app_gradient=None, empirical_mean=None, lambdas=None):
        """
        :param m: a row matrix of values for m
        :param step: 
        :param decay: 
        :param speed: 
        :param estimated_mean
        :param app_gradient
        :param empirical_mean
        :param lambdas
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :return: 
        """
        pre_m = deepcopy(m)
        if estimated_mean is None:
            est_mean, mu = Regression.get_est_mean(m, data_matrix)
        else:
            est_mean = estimated_mean
        if app_gradient is None:
            if empirical_mean is None:
                empirical_mean = Regression.get_empirical_mean(data_matrix)
            app_gradient, avg_error = Regression.get_update_value_and_error(
                    est_mean, empirical_mean, m, lambdas)
        else:
            app_gradient = app_gradient
        tmp1 = app_gradient.tolist()[0]
        m = m.tolist()[0]
        for i in range(0, pre_m.shape[1]):
            m[i] = m[i] + step * decay * speed * tmp1[i]
        m = np.matrix(m, dtype=np.float64)
        diff_m_yy = m[0, 0] - pre_m[0, 0]
        lower_bound = Regression.get_m_yy_lower_bound(data_matrix)
        while m[0, 0] < lower_bound:
            m[0, 0] += abs(diff_m_yy)/2.0
        return m

    @staticmethod
    def get_m_yy_lower_bound(data_matrix):
        """
        get lower bound for Myy
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :return: lower bound of M_yy
        """
        matrix = data_matrix[:, :-1]
        y_col = data_matrix[:, -1]
        weight = Regression.cal_weight(matrix)
        _, base_variance = Regression.get_dist_base(y_col)
        lower_bound = -1 / (2 * base_variance * weight[0])
        for i in range(1, len(weight)):
            temp = -1.0 / (2 * base_variance * weight[i])
            if temp > lower_bound:
                lower_bound = temp
        return lower_bound

    @staticmethod
    def start(init_m, data_matrix, stop_thd, rate_initial, decay_tune, 
              iteration=500):
        """
        start training process
        :param init_m: a row matrix of initial values of m
        :param data_matrix: M*(N+1) matrix, M - number of instances, N - number of features, 
        the last column is target value
        :param stop_thd: 
        :param rate_initial: 
        :param decay_tune: 
        :param iteration: 
        :return: 
        """
        n = 0
        m_, n_ = data_matrix.shape
        standard_lambda = 1 / math.sqrt(m_)
        lambdas = standard_lambda
        empirical_mean = Regression.get_empirical_mean(data_matrix)
        m = init_m
        rate = rate_initial
        speed = 1
        decay = 1
        pre_avg_error = 1e20
        min_avg_error = 1e20
        current_m = deepcopy(m)
        best_m = None
        while True:
            logger.info("*"*10+str(n)+"*"*10)
            n = n + 1
            if n > iteration:
                return best_m
            estimated_mean, mu = Regression.get_est_mean(current_m, data_matrix)
            app_gradient, avg_error = Regression.get_update_value_and_error(
                estimated_mean,
                empirical_mean,
                current_m,
                lambdas
            )
            logger.info("average error: %s" % (avg_error,))
            if abs(pre_avg_error - avg_error) < 1e-15:
                if min_avg_error < 10 * stop_thd:
                    return best_m
            if avg_error < min_avg_error:
                if np.count_nonzero(current_m) == 0:
                    pass
                else:
                    min_avg_error = avg_error
                    best_m = current_m
            current_m = Regression.update_m(
                current_m,
                rate, decay,
                speed,
                data_matrix,
                lambdas=lambdas,
                estimated_mean=estimated_mean,
                app_gradient=app_gradient,
                empirical_mean=empirical_mean
            )
            decay = decay_tune / (decay_tune + math.sqrt(n))
            pre_avg_error = avg_error

    @staticmethod
    def get_true_y(target):
        y = target[:, -1]
        return y.transpose().tolist()[0]

    @staticmethod
    def get_prediction_error(prediction_value, true_value):
        error = 0
        for i in range(len(prediction_value)):
            error += abs(prediction_value[i]-true_value[i])
        return error/len(prediction_value)

# if __name__ == '__main__':
#     regression = Regression()
#     #matrix_ = Regression.read_csv('../GGsrc', size=None)
#     matrix_ = Regression.read_csv('../pm2.5_srcFile.csv', size=400)
#     m_size, n_size = matrix_.shape
#     m0_ = np.matrix(np.zeros(1 + n_size), dtype=np.float64)
#     stopThd = 1e-5
#     rateInitial = 0.01
#     decayTune = 0.01
#     iteration = 1000
#     # find execution time
#     start_time = time.clock()
#     best_m_ = Regression.start(
#         m0_, matrix_, stop_thd=stopThd, rate_initial=rateInitial, decay_tune=decayTune, iteration=iteration)
#     end_time = time.clock()
#     print "Execution time for %d iterations is: %s min" % (
#         iteration, (end_time-start_time)/60.0)
#     print "train done"
#     #target_ = Regression.read_csv('../GGtarget', size=None)
#     target_ = Regression.read_csv('../pm2.5_trgFile.csv', size=10000)
#     _, predict_y = Regression.get_est_mean(best_m_, target_)
#     true_y = Regression.get_true_y(target_)
#     trr = Regression.get_prediction_error(predict_y, true_y)
#     print "target avg error size pm2.5_f400:", trr
