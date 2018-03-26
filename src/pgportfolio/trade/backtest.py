from __future__ import absolute_import, division, print_function
import numpy as np
from pgportfolio.trade import trader
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
from pgportfolio.tools.trade import calculate_pv_after_commission


class BackTest(trader.Trader):
    def __init__(self, config, net_dir=None, agent=None, agent_type="nn", history=None, window_size=50, trading_commission=0.0025):
        config["input"]["feature_number"] = 1
        config["input"]["coin_number"] = history.shape[0]
        config["input"]["window_size"] = window_size
        config["trading"]["trading_consumption"] = trading_commission
        trader.Trader.__init__(self, 0, config, 0, net_dir,
                               initial_BTC=1, agent=agent, agent_type=agent_type)
        if agent_type == "nn":
            data_matrices = self._rolling_trainer.data_matrices
        elif agent_type == "traditional":
            pass
        else:
            raise ValueError()
        # history is shape [num_assets, num_periods, 4]
        self.history = history[:, :, 3, None]
        self.history = np.transpose(self.history, [2, 0, 1])
        self.test_indices = np.arange(self.history.shape[2])[:-(window_size+1)]
        self.__test_set = self.get_test_set()
        self.__test_length = self.__test_set["X"].shape[0]
        self._total_steps = self.__test_length
        self.__test_pv = 1.0
        self.__test_pc_vector = []
        self._agent.test_relative = self.get_test_relative()

    @property
    def test_pv(self):
        return self.__test_pv

    @property
    def test_pc_vector(self):
        return np.array(self.__test_pc_vector, dtype=np.float32)

    def finish_trading(self):
        self.__test_pv = self._total_capital

        """
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(self._rolling_trainer.data_matrices.sample_count)),
               self._rolling_trainer.data_matrices.sample_count)
        fig.tight_layout()
        plt.show()
        """

    def get_test_set(self):
        return self.__pack_samples(self.test_indices)

    def __pack_samples(self, indexs):
        indexs = np.array(indexs)

        M = [self.get_submatrix(index) for index in indexs]
        M = np.array(M)
        X = M[:, :, :, :-1] # [num_test_periods, features, num_assets, window_size]
        y = M[:, :, :, -1] / M[:, 0, None, :, -2] # [num_test_periods, features, num_assets]
        return {"X": X, "y": y}

    def get_submatrix(self, ind):
        #  global data has [features, num_assets, window_size+1]
        return self.history[:, :, ind:ind+self._window_size+1]

    def _log_trading_info(self, time, omega):
        pass

    def _initialize_data_base(self):
        pass

    def _write_into_database(self):
        pass

    def __get_matrix_X(self):
        return self.__test_set["X"][self._steps]

    def __get_matrix_y(self):
        return self.__test_set["y"][self._steps, 0, :]

    def get_test_relative(self):
        test_set = self.__test_set["y"][:, 0, :].T
        test_set = np.concatenate((np.ones((1, test_set.shape[1])), test_set), axis=0)
        return test_set

    def rolling_train(self, online_sample=None):
        self._rolling_trainer.rolling_train()

    def generate_history_matrix(self):
        inputs = self.__get_matrix_X()
        if self._agent_type == "traditional":
            inputs = np.concatenate([np.ones([1, 1, inputs.shape[2]]), inputs], axis=1)
            inputs = inputs[:, :, 1:] / inputs[:, :, :-1]
        return inputs

    def trade_by_strategy(self, omega):
        logging.info("the step is {}".format(self._steps))
        logging.debug("the raw omega is {}".format(omega))
        future_price = np.concatenate((np.ones(1), self.__get_matrix_y()))
        pv_after_commission = calculate_pv_after_commission(omega, self._last_omega, self._commission_rate)
        portfolio_change = pv_after_commission * np.dot(omega, future_price)
        self._total_capital *= portfolio_change
        self._last_omega = pv_after_commission * omega * \
                           future_price /\
                           portfolio_change
        logging.debug("the portfolio change this period is : {}".format(portfolio_change))
        self.__test_pc_vector.append(portfolio_change)

