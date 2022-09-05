import os
from itertools import chain

from setuptools import setup
from setuptools.config import read_configuration


extras = read_configuration("setup.cfg")['options']['extras_require']

# # Dev is everything
# extras['dev'] = list(chain(*extras.values()))

# # All is everything but tests and docs
# exclude_keys = ("tests", "docs", "dev")
# ex_extras = dict(filter(lambda i: i[0] not in exclude_keys, extras.items()))
# # Concatenate all the values together for 'all'
# extras['all'] = list(chain.from_iterable(ex_extras.values()))

setup(extras_require=extras,
include_package_data=True,
data_files=[
        ('optimus/material/',['optimus/material/Material_database.xls',
        'optimus/material/Material_database_user-defined.xls']),
    ]
    )
