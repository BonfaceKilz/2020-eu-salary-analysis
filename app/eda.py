# eda.py ---

# Copyright (C) 2022  Munyoki Kilyungi <bonface.kilyungi@strathmore.edu>
# Copyright (C) 2022  Gertrude Gichuhi <ggichuhi@strathmore.edu>
# Copyright (C) 2022  Eunice Wanyoike <eunice.wanyoike@strathmore.edu>
# Copyright (C) 2022  Peter Mburu <nyambura.peter@strathmore.edu>
# Copyright (C) 2022  Beatrice Ngeno <beatrice.ngeno@strathmore.edu>

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


import pandas as pd
from typing import Tuple


def fix_outliers(
    dataframe: pd.DataFrame,
    feature: str,
    q1: int,
    q3: int,
    n: float | int = 1.5
) -> Tuple[int, pd.DataFrame]:
    """Remove outliers from a dataframe using the Interquartile Range
    (IQR) technique"""
    df = dataframe.copy()
    p25, p75 = df[feature].quantile(q1), df[feature].quantile(q3)
    iqr = p75 - p25
    upper_limit, lower_limit = p75 + n * iqr, p25 - n * iqr
    # Removing the outliers
    _df = df[(df[feature] > lower_limit) & (df[feature] < upper_limit)]
    return (df.shape[0] - _df.shape[0], _df)
