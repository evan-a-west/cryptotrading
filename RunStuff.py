import BuildColumns as bc
from dateutil.parser import parse

if __name__ == "__main__":
    stuff = bc.BuildColumns(
        startDate=parse('2020-9-23'),
        endDate=parse('2021-1-31'),
        # endDate=parse('2020-9-30'),
        coins=["BCH"])

    # stuff.readBuildSave()
    stuff.readSplitSave("processedData2021-04-12.csv")
