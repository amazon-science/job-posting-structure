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
