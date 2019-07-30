import numpy as np
import copy
from glue.config import data_factory
from glue.core import Data, Component
from casatools import table

def is_measurement_set(filename, **kwargs):
    return filename.endswith('.ms') or filename.endswith('.ms/')

@data_factory('MS loader', is_measurement_set)
def read_measurement_set(file_name):
    tb = table()
    tb.open(file_name)
    columns = tb.colnames()
    columns.remove('DATA')
    datadict = {}
    fulldata = MeasurementSetComponent(file_name, 'DATA')
    realdata = copy.copy(fulldata)
    realdata._data = fulldata._data.real
    imagdata = copy.copy(fulldata)
    imagdata._data = fulldata._data.imag
    datadict['REAL_DATA'] = realdata
    datadict['IMAG_DATA'] = imagdata
    datashape = datadict['REAL_DATA'].data.shape
    for cn in columns:
        try:
            datadict[cn] = MeasurementSetComponent(file_name, cn, shape=datashape)
        except RuntimeError:
            # data do not exist or can't be read for some reason
            pass
        except ValueError:
            print(f"FAILED to load {cn}")
    result = CASAData(**datadict)
    tb.close()
    return result

class CASAData(Data):
    pass

class MeasurementSetComponent(Component):
    def __init__(self, filename, colname, shape=None):
        self.filename = filename
        self.colname = colname

        # gotta be a better way, right?
        if not hasattr(self, '_data'):
            tb = table()
            tb.open(self.filename)
            self._data = tb.getcol(self.colname)
            if shape is not None:
                self._data = np.broadcast_to(self._data, shape)
            tb.close()

    @property
    def data(self):
        if not hasattr(self, '_data'):
            tb = table()
            tb.open(self.filename)
            self._data = tb.getcol(self.colname)
            tb.close()

        return self._data

    def __getitem__(self, key):
        logging.debug("Using %s to index data of shape %s", key, self.shape)
        return self.data[key]

    @property
    def numeric(self):
        return True
