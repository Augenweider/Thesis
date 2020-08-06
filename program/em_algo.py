class em_algo:
    """
    An EM algorithm finds the fitted parameters for the inference model or
    the model theory trained by the data set given a list of initial parameters.
    """
    def __init__(self, initial_theta, observations):
        """
        The initial function of the EM algorithm.

        Args:
            initial_theta(list): a list of initial parameters
            observations(list): a list of samples
        """
        assert len(initial_theta) == 3 or len(initial_theta) == 5
        self.prior = initial_theta
        self.observations = observations
        if len(initial_theta) == 3:
            self.type = "insight"
            self.c = initial_theta[0]
            self.e = initial_theta[1]
            self.f = initial_theta[2]
        elif len(initial_theta) == 5:
            self.type = "inference"
            self.c = initial_theta[0]
            self.d = initial_theta[1]
            self.x = initial_theta[2]
            self.s = initial_theta[3]
            self.i = initial_theta[4]

    def em_single(self, theta_list, obser, param=None):
        """
        One iteration step of the EM algorithm which contains E-step and M-step.
        And return the trained parameters.

        Args:
            theta_list(list): the given list of parameters
            obser(list): one of the samples
        """
        if self.type == "insight":
            # E-step
            for (index, j) in enumerate(obser):
                if index == 0:
                    m_0 = j
                elif index == 3:
                    m_3 = j
                elif index == 1:
                    p_1_1 = (1 - self.c) * self.e * (1 - self.f)
                    p_2_1 = self.c * (1 - self.e)
                    p_1 = p_1_1 + p_2_1
                    m_1_1 = j * p_1_1 / p_1
                    m_2_1 = j * p_2_1 / p_1
                elif index == 2:
                    p_1_2 = (1 - self.c) * self.e * self.f
                    p_2_2 = self.c * self.e * self.f
                    p_2 = p_1_2 + p_2_2
                    if p_2 == 0:
                        m_1_2 = 0
                        m_2_2 = 0
                    else:
                        m_1_2 = j * p_1_2 / p_2
                        m_2_2 = j * p_2_2 / p_2
            # M-step
            total = (m_0 + m_1_1 + m_2_1 + m_1_2 + m_2_2 + m_3)
            self.c = (m_2_1 + m_2_2 + m_3) / total
            self.e = (m_1_1 + m_1_2 + m_2_2 + m_3) / total
            self.f = (m_1_2 + m_2_2) / total
            if param == 'c':
                self.c = 0
            elif param == '-c':
                self.c = 1
            elif param == 'e':
                self.e = 0
            elif param == '-e':
                self.e = 1
            elif param == 'f':
                self.f = 0
            elif param == '-f':
                self.f = 1
            return [self.c, self.e, self.f]

        if self.type == "inference":
            dic = {}
            # E-step
            for (index, j) in enumerate(obser):
                if (index == 1 or index == 2 or index == 3 or index == 4 or index == 5
                        or index == 8 or index == 10 or index == 12):
                    exec("dic['m_%d'] = %d" % (index, j))
                elif index == 6:
                    p_1_6 = self.c * self.d * (1 - self.s) * (1 - self.i)
                    p_2_6 = self.c * (1 - self.d) * self.s * (1 - self.i)
                    p_6 = p_1_6 + p_2_6
                    if p_6 == 0:
                        dic['m_1_6'] = 0
                        dic['m_2_6'] = 0
                    else:
                        dic['m_1_6'] = j * p_1_6 / p_6
                        dic['m_2_6'] = j * p_2_6 / p_6
                elif index == 9:
                    p_1_9 = self.c * self.d * self.s * (1 - self.i)
                    p_2_9 = self.c * (1 - self.d) * (1 - self.s) * (1 - self.i)
                    p_9 = p_1_9 + p_2_9
                    if p_9 == 0:
                        dic['m_1_9'] = 0
                        dic['m_2_9'] = 0
                    else:
                        dic['m_1_9'] = j * p_1_9 / p_9
                        dic['m_2_9'] = j * p_2_9 / p_9
                elif index == 15:
                    p_1_15 = (1 - self.c) * self.x * self.s * (1 - self.i)
                    p_2_15 = (1 - self.c) * self.x * (1 - self.s) * (1 - self.i)
                    p_3_15 = (1 - self.c) * (1 - self.x) * self.d * (1 - self.i)
                    p_4_15 = (1 - self.c) * (1 - self.x) * (1 - self.d) * (1 - self.i)
                    p_15 = p_1_15 + p_2_15 + p_3_15 + p_4_15
                    if p_15 == 0:
                        dic['m_1_15'] = 0
                        dic['m_2_15'] = 0
                        dic['m_3_15'] = 0
                        dic['m_4_15'] = 0
                    else:
                        dic['m_1_15'] = j * p_1_15 / p_15
                        dic['m_2_15'] = j * p_2_15 / p_15
                        dic['m_3_15'] = j * p_3_15 / p_15
                        dic['m_4_15'] = j * p_4_15 / p_15
                else:
                    exec("dic['m_%d'] = 0" % index)
            # M-step
            total = sum(dic.values())
            self.c = (dic['m_1'] + dic['m_2'] + dic['m_4'] + dic['m_1_6'] + dic['m_2_6'] + dic['m_8'] + dic['m_1_9'] + dic['m_2_9']) / total
            self.d = (dic['m_8'] + dic['m_1_9'] + dic['m_4'] + dic['m_1_6'] + dic['m_12'] + dic['m_3_15']) / total
            self.x = (dic['m_10'] + dic['m_1_15'] + dic['m_5'] + dic['m_2_15']) / total
            self.s = (dic['m_8'] + dic['m_1_9'] + dic['m_2'] + dic['m_2_6'] + dic['m_7'] + dic['m_1_15']) / total
            self.i = (dic['m_8'] + dic['m_4'] + dic['m_2'] + dic['m_1'] + dic['m_10']
                      + dic['m_5'] + dic['m_12'] + dic['m_3']) / total

            if param == 'c':
                self.c = 0
            elif param == '-c':
                self.c = 1
            elif param == 'd':
                self.d = 0
            elif param == '-d':
                self.d = 1
            elif param == 'x':
                self.x = 0
            elif param == '-x':
                self.x = 1
            elif param == 's':
                self.s = 0
            elif param == '-s':
                self.s = 1
            elif param == 'i':
                self.i = 0
            elif param == '-i':
                self.i = 1
            return [self.c, self.d, self.x, self.s, self.i]

    def run(self, parameter=None):
        """
        Iterate the single steps of the EM algorithm and return fitted parameters.
        """
        for obser in self.observations:
            if self.type == "insight":
                if parameter == 'c':
                    self.c = 0
                elif parameter == '-c':
                    self.c = 1
                elif parameter == 'e':
                    self.e = 0
                elif parameter == '-e':
                    self.e = 1
                elif parameter == 'f':
                    self.f = 0
                elif parameter == '-f':
                    self.f = 1
            else:
                if parameter == 'c':
                    self.c = 0
                elif parameter == '-c':
                    self.c = 1
                elif parameter == 'd':
                    self.d = 0
                elif parameter == '-d':
                    self.d = 1
                elif parameter == 'x':
                    self.x = 0
                elif parameter == '-x':
                    self.x = 1
                elif parameter == 's':
                    self.s = 0
                elif parameter == '-s':
                    self.s = 1
                elif parameter == 'i':
                    self.i = 0
                elif parameter == '-i':
                    self.i = 1
            self.prior = self.em_single(self.prior, obser, parameter)
        return self.prior
