import yaml


class Config(dict):
    def __init__(self, filename, mode):
        super().__init__()
        config_file = "./configs/{:s}.yaml".format(filename)
        with open(config_file, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['MODE'] = mode

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        else:
            return None

    def print_info(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------\n')
