# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
try:
    import eyelink_data_reader
    import phone_data_reader
except:
    from . import eyelink_data_reader
    from . import phone_data_reader

def split_data(data_source='phone', random_seed=42):
    if data_source == 'phone':
        return phone_data_reader.split_data(random_seed=random_seed)
    elif data_source == 'eyelink':
        return eyelink_data_reader.split_data(random_seed=random_seed)
    else:
        raise ValueError('data_source must be either "phone" or "eyelink"')