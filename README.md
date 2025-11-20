# CSINTSY MCO3 Catbot

_A simple cat‑catching bot using reinforcement learning_

## 🚀 Project Overview

The goal is to build an intelligent agent (the
"Catbot") that learns to catch a cat in a simulated environment using
reinforcement learning techniques.

Key points:\

- The environment simulates a "cat" moving around and the agent's goal
  is to catch it.\
- The agent is trained using reinforcement learning algorithms in
  `training.py`.\
- `bot.py` contains the agent logic.\
- `cat_env.py` defines the custom environment.\
- `play.py` allows running the trained agent.\
- Utility functions are in `utility.py`.

## 📁 Repository Structure

    /
    ├── bot.py
    ├── cat_env.py
    ├── training.py
    ├── play.py
    ├── utility.py
    ├── requirements.txt
    ├── specifications.pdf
    ├── images/
    └── LICENSE

## 🧠 Getting Started

### Prerequisites

- Python 3.x\
- Virtual environment recommended

### Installation

    git clone https://github.com/mbchavez27/csintsy-mco3-catbot.git
    cd csintsy-mco3-catbot
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

### Training

    python training.py

### Run Agent

    python play.py
