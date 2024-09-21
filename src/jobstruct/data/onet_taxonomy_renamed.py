# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright National Association of State Workforce Agencies. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

import csv
import json

skills = []

with open("onet_taxonomy_renamed.csv") as f:
    for row in csv.DictReader(f):
        if row["Renamed"]:
            skills += row["Renamed"].split(",")

with open("onet_taxonomy_renamed.json", "w") as f:
    json.dump(
        {
            "name": "Skills",
            "children": [{"name": skill} for skill in skills]
        },
        f,
        indent=2
    )
