import BuildColumns as bc
from dateutil.parser import parse

if __name__ == "__main__":
    stuff = bc.BuildColumns(
        startDate=parse('2020-9-23'),
        endDate=parse('2020-10-31'),
        coins=["BCH"])

    stuff.readBuildSave()
