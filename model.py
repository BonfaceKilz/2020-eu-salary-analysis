# model.py ---

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
import numpy as np
import funcy as func
from sklearn.ensemble import RandomForestRegressor as Estimator
from sklearn.model_selection import train_test_split
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from typing import Any
from app.remaps import COLUMNS_REMAP
from app.remaps import PL_REMAP
from app.eda import fill_missing_values
from app.eda import fix_outliers
from app.eda import reduce_dimension_position
from app.eda import reduce_dimension_seniority
from app.eda import sanitize_years_of_experience


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    _df.rename(columns=COLUMNS_REMAP, inplace=True)
    _df.rename(columns=PL_REMAP, inplace=True)
    _df = func.rcompose(
        func.autocurry(fix_outliers)(feature="Salary (2020)",
                                     q1=0.25, q3=0.75, n=0.1),
        func.second,
        fill_missing_values,
        reduce_dimension_position,
        reduce_dimension_seniority,
        sanitize_years_of_experience,
    )(_df)
    _df["Company size"] = df["Company size"].replace(
        {
            "up to 10": 1,
            "11-50": 2,
            "51-100": 3,
            "101-1000": 4,
            "1000+": 5,
        }
    )
    return _df


def build_model(df: pd.DataFrame) -> Any:
    _df = df.copy()
    _df["LogSalary (2020)"] = np.log2(_df["Salary (2020)"])
    gender_df = pd.get_dummies(_df["Gender"])
    position_df = pd.get_dummies(_df["Position"])
    _df = pd.concat([_df, gender_df, position_df], axis=1)
    x_vars = func.select(
        func.rpartial(isinstance, str),
        [
            "Age",
            "Female",
            "Other (Gender)",
            "Company size",
            "Salary (2019)",
            *_df["Position"].unique().tolist(),
        ],
    )
    _df = _df.dropna(subset=["LogSalary (2020)", *x_vars])

    X_train, X_test, y_train, y_test = train_test_split(
        _df[x_vars], _df["LogSalary (2020)"]
    )
    model = Estimator()
    model.fit(X_train, y_train)
    return RegressionExplainer(model, X_test, y_test)


if __name__ == "__main__":
    ExplainerDashboard(
        build_model(clean_data(pd.read_csv("2020-eu-salary.csv")))
    ).run()
