# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

import jobstruct
import json
import os
import sys
from pathlib import Path

DIR = Path(os.path.realpath(os.path.dirname(__file__)))

def test_sde_amazon_robotics():
    j = jobstruct.JobStructHTML.from_file(DIR / "SDE_Amazon_Robotics.html")
    print(j)
    with open(DIR / "SDE_Amazon_Robotics_JobStructHTML.json") as f:
        true_dict = json.load(f)
    assert j.to_dict() == true_dict

def reset_test_output():
    """
    Reset test output files.
    """
    j = jobstruct.JobStructHTML.from_file(DIR / "SDE_Amazon_Robotics.html")
    with open(DIR / "SDE_Amazon_Robotics_JobStructHTML.json", "w") as f:
        json.dump(j.to_dict(), f)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_test_output()
