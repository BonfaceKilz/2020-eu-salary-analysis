# Introduction

This is repository contains some group work we did for my Msc. Class.
Here, we analyze the EU 2020 dataset with the following goals:

- Relevant insights for some one who wants to get into the industry.
- Dashboard to show career path.
- Telling a good story

Here are a set of questions that helped us when exploring our dataset:

- Do years of experiences matter?
- Are there outliers?
- What sectors are important?
- Does Education matter? -- can't answer this
- What PLs matter when it comes to enumeration?  Can you say how?
- Does seniority matter?


# Getting Started

Assumptions: you are running python 3.10 or later.  Also make sure you have virtualenv installed.

### Installation

- Clone the repo
- Create your virtual environment in the root of your project:

```bash
virtualenv --python python3.10 .venv
```

- Activate your virtual environment (the following steps assume that you'll be working from your environment) and Install dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

- Install your ".venv" as a kernel to your notebook:

```bash
ipython kernel install --user --name=.venv
```

- To generate the "models" dashboard run:

```bash
python model.py
```

#### GNU/Guix Users

Setting this up with GNU/Guix is easier:

```
guix shell python python-notebook -N bash -- \
           ipython kernel install --user --name=.venv
```

Running the notebook:

```
guix shell python python-notebook -N bash -- \
           jupyter notebook --ip=0.0.0.0 --port=8080
```


