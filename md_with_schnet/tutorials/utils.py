import platform

def set_data_prefix() -> str:
    """
    Set the data prefix depending on the system.

    Returns:
        str: Data prefix path.
    """
    if platform.system() == 'Darwin':
        return '/Users/ferdinandtolkes/whk/data/'
    elif platform.system() == 'Linux':
        return '/loctmp/tof54964/data/'
    else:
        raise ValueError('Unknown system. Please set data_prefix manually.')
