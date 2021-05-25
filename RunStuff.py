import DataProcessing as bc
from dateutil.parser import parse

if __name__ == "__main__":
    stuff = bc.BuildColumns(
        startDate=parse('2020-9-23'),
        endDate=parse('2021-4-15'),
        # endDate=parse('2020-10-15'),
        coins=["DOGE"])
    # coins=["BCH", "BSV", "BTC", "BTG", "DASH", "DOGE", "ETC", "ETH", "LSK", "LTC", "NEO", "OMG", "QTUM", "XLM", "XMR", "XRP", "ZEC"])
    stuff.Step1_ReadBuildSave()
    # stuff.readSplitSave("processedData2021-04-12.csv")
