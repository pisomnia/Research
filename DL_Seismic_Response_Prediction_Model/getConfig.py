import configparser

def parse_int_tuple(int_tuple):
    return tuple(int(k.strip()) for k in int_tuple[1:-1].split(','))

def get_config(config_file='config.ini'):
    parser = configparser.ConfigParser()
    parser.read(config_file)
    # get the ints, floats and strings
    _conf_ints = [(key, int(value)) for key, value in parser.items('ints')]
    _conf_floats = [(key, float(value)) for key, value in parser.items('floats')]
    _conf_strings = [(key, str(value)) for key, value in parser.items('strings')]
    _conf_tuples = [(key, parse_int_tuple(value)) for key, value in parser.items('tuples')]

    print(_conf_tuples)
    return dict(_conf_ints + _conf_floats + _conf_strings + _conf_tuples)


