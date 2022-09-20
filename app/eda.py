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
from app.remaps import PL_REMAP
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


def reduce_dimension_seniority(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reduce the number of dimensions in the 'Seniority level'
    column to intepretable seniority levels that are more general"""
    _remap = {}
    _df = dataframe.copy()
    for col in _df["Seniority level"].unique():
        match col:
            case "Head":
                _remap[col] = "Lead"
            case col if any(
                [
                    col in ["VP", "No level", "Director", "Key"],
                    "manager" in str(col).lower(),
                ]
            ):
                _remap[col] = "Manager"
            case col if any(
                [
                    col in ["Principal", "No level "],
                    "no idea" in str(col).lower(),
                ]
            ):
                _remap[col] = "Senior"
            case col if col in ["Entry level", "Intern"]:
                _remap[col] = "Junior"
    _df["Seniority level"].replace(
        _remap.keys(), _remap.values(), inplace=True
    )
    return _df


def reduce_dimension_position(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Reduced the dimentionality of company positions"""
    _remap = {}
    _df = dataframe.copy()
    for col in _df["Position"].unique():
        match col:
            case col if "lead" in str(col).lower():
                _remap[col] = "Team Lead"
            case col if "ios" in str(col).lower():
                _remap[col] = "Mobile Developer"
            case col if col in ["QA Engineer"] or "test" in (
                c := str(col).lower()
            ) or "qa" in c:
                _remap[col] = "QA Engineer"
            case col if col in [
                "DatabEngineer",
                "data engineer",
                "Big Data Engineer",
                "Senior Data Engineer",
            ]:
                _remap[col] = "Data Engineer"
            case col if any(
                [
                    col
                    in [
                        "Stuttgart",
                        "Recruiter",
                        "Consultant",
                        "Presales Engineer",
                        "Researcher",
                        "Localization producer",
                        "Reporting Engineer",
                        "agile master",
                        "Banker",
                        "Agile Coach",
                        "Scrum Master",
                        "Beikoch",
                        "It Consulting",
                        "Computational linguist",
                        "Rentner",
                        "Application Consultant",
                        "Professor",
                    ],
                    "sales" in (c := str(col).lower()),
                    "consult" in c,
                    "agile" in c,
                    "student" in c,
                    "recruit" in c,
                ]
            ):
                _remap[col] = "Other (Position)"
            case col if col in [
                "Fullstack Developer",
                "IT Spezialist",
                "Embedded Software Engineer",
                "Sofware/Hardware Engineer",
                "Software Engineer",
                "Firmware Engineer",
                "Hardware Engineer",
            ] or "full" in (
                c := str(col).lower()
            ) or "java" in c or "data engineer" in c:
                _remap[col] = "Software/Hardware Engineer"
            case col if any(
                [
                    "head" in (c := str(col).lower()),
                    "manage" in c,
                    "scrum" in c,
                    "cto" in c,
                    "vp" in c,
                    "director" in c,
                ]
            ):
                _remap[col] = "Manager"
            case col if "insights" in (
                c := str(col).lower()
            ) or "analyst" in c or "analytics" in c:
                _remap[col] = "Data Analyst"
            case col if col in [
                "DevOps",
                "SRE",
                "DBA",
                "Support Engineer",
                "support engineer",
            ] or any(
                [
                    "security" in (c := str(col).lower()),
                    "roboti" in c,
                    "sap" in c,
                    "system" in c,
                    "cloud" in c,
                    "network" in c,
                ]
            ):
                _remap[col] = "Infra"
            case col if "archite" in str(col).lower():
                _remap[col] = "Architect"
            case col if col in ["Designer (UI/UX)", "Graphic Designer"]:
                _remap[col] = "UI/UX"
    _df["Position"].replace(
        _remap.keys(), _remap.values(), inplace=True
    )
    return _df


def sanitize_years_of_experience(dataframe: pd.DataFrame) -> pd.DataFrame:
    _df = dataframe.copy()
    for error, fix in (
        ("6 (not as a data scientist, but as a lab scientist)", "6"),
        ("less than year", "1"),
        ("15, thereof 8 as CTO", "23"),
        ("1 (as QA Engineer) / 11 in total", "12"),
        ("1,5", "1.5"),
        ("2,5", "2.5"),
    ):
        _df["Experience (Years)"] = _df["Experience (Years)"].replace(
            to_replace=error, value=fix
        )
    _df["Experience (Years)"] = _df["Experience (Years)"].fillna(_df["Experience (Years)"].median())
    # Convert everything to a float
    _df["Experience (Years)"] = _df["Experience (Years)"].astype(float)
    return _df


def fill_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    _df = dataframe.copy()
    # Using the median to fill in the missing values:
    _df["Position"] = _df["Position"].fillna("Other (Position)")
    _df["Gender"] = _df["Gender"].fillna("Other (Gender)")
    _df["Age"] = _df["Age"].fillna(_df["Age"].median())
    _df["PL"] = _df["PL"].fillna("Other (PL)")
    _df["Company size"] = _df["Company size"].fillna("1000+")
    _df['PL'] = _df['PL'].replace(PL_REMAP)
    return _df
