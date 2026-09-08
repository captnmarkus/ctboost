import json
import warnings
from pathlib import Path

import ctboost
import numpy as np
import pandas as pd


class Handler:
    def __init__(self):
        self.events = []

    def __call__(self, error, flag):
        self.events.append([error, flag])

    def write(self, message):
        self.events.append(message)


records = []
pipeline = ctboost.FeaturePipeline().fit(np.zeros((3, 1)), [0, 1, 2], feature_names=['value'])
for dtype, bits in [('f2', 0x7C01), ('f4', 0x7F800001), ('f8', 0x7FF0000000000001)]:
    values = np.array([bits] * 16, dtype=dtype.replace('f', 'u')).view(dtype).reshape(16, 1)
    for kind in ['numpy', 'native', 'pandas']:
        data = pd.DataFrame(values, columns=['value']) if kind == 'pandas' else values
        for policy in ['warn', 'call', 'log']:
            outcomes = {}
            for arm in ['object_reference', 'typed']:
                handler = Handler()
                previous_handler = np.geterrcall()
                np.seterrcall(handler)
                try:
                    with warnings.catch_warnings(record=True) as caught, np.errstate(invalid=policy):
                        warnings.simplefilter('always')
                        try:
                            if arm == 'object_reference':
                                boxed = data.to_numpy(dtype=object) if kind == 'pandas' else np.asarray(values, dtype=object)
                                result = pipeline._native.transform_array(boxed, ['value'])[0]
                            elif kind == 'native':
                                result = pipeline._native.transform_array(values, ['value'])[0]
                            else:
                                result = pipeline.transform_array(data, feature_names=['value'])[0]
                            outcome = {'bits': result.view(np.uint32).tolist()}
                        except Exception as error:
                            outcome = {'error': type(error).__name__, 'message': str(error)}
                    outcome['warnings'] = [str(item.message) for item in caught]
                    outcome['events'] = handler.events
                    outcomes[arm] = outcome
                finally:
                    np.seterrcall(previous_handler)
            records.append({'dtype': dtype, 'kind': kind, 'policy': policy, 'equal': outcomes['object_reference'] == outcomes['typed'], **outcomes})
report = {'ctboost': ctboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'records': records}
print(json.dumps(report, indent=2))
