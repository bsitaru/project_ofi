
class Summary:
    def __init__(self, file_path: str, model_name: str):

        variables = int(model_name.split('_')[1])
        if model_name.startswith('SplitOFI'):
            variables = variables * 3
        variables += 1

        with open(file_path, 'r') as f:
            # Title and separator
            for _ in range(2):
                f.readline()

            def get_2_elems():
                l = f.readline()
                l = l.split('   ')
                l = list(filter(lambda x: len(x) > 0, l))
                return l[1].strip(), l[3].strip()

            def get_1_elem():
                l = f.readline()
                l = l.split('   ')
                l = list(filter(lambda x: len(x) > 0, l))
                return l[1].strip()

            self.dep_variable, self.r_squared = get_2_elems()
            self.model, self.adj_r_squared = get_2_elems()
            self.method, self.f_stat = get_2_elems()
            self.date, self.prob_f_stat = get_2_elems()
            self.time, self.log_likelihood = get_2_elems()
            self.no_obs, self.aic = get_2_elems()
            self.df_residuals, self.bic = get_2_elems()
            self.df_model = get_1_elem()
            self.cov_type = get_1_elem()

            for _ in range(3):
                f.readline()

            self.vars = []
            for i in range(variables):
                l = f.readline()
                l = l.split('  ')
                l = list(filter(lambda x: len(x) > 0, l))
                d = {}
                d['name'] = l[0].strip()
                d['coef'] = float(l[1].strip())
                d['std_err'] = float(l[2].strip())
                d['t'] = float(l[3].strip())
                d['p>t'] = float(l[4].strip())
                d['0.025'] = float(l[5].strip())
                d['0.975'] = float(l[6].strip())
                self.vars.append(d)

            l = f.readlines()
            self.oos_r_squared = float(l[-1].split(' ')[-1])

