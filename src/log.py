import pandas as pd

class Logger():

    def __init__(self, columns):
        self.log = pd.DataFrame(columns=columns)


    def add_log(self, index, log_content):
        assert len(self.log.columns) == len(log_content)
        self.log.loc[index] = log_content


    def save(self, path):
        self.log.to_csv(path, float_format="%.8f")


if __name__ == "__main__":
    pass
