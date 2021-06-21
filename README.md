# vc-analysis-tools
Simple data analysis tools for ventures investments.

## [analyze_rounds.py](analyze.py) - NLP x Funding Rounds
Examines funding rounds exported from CrunchBase in CSV format. Using NLP clusters the investment rounds
by industry similarity. 

The only 2 features of this application are:
1. generation of a network graph of the funding rounds
2. displaying of a correlation matrix

To download the data (subscription required), go to [crunchbase pro > search > funding rounds](https://www.crunchbase.com/discover/funding_rounds)
and filter to the data of interest. Then select the "Export to CSV" button on the top-right, and save
the .csv file in the folder of this application.

Example with Funding Rounds from Jan 1 to Jun 14 where Tiger Global participated as investor:
![Tiger Global example](examples/Tiger%206.14.2021.png)

You will notice that there are clusters of related investments, for instance in healthcare, crypto/finance, and data/ops/rpa.
![Tiger Global example](examples/Tiger%206.14.2021%20-%20Annotated.png)

### NLP - tech background
This works by computing the 'sentence similarity' of one startup's "Industry" versus everyone else.

For practitioners, this is a two steps process:
1. Industries (N strings) -> Embeddings (N * 512 embedding values)
2. Inner-product of Embeddings -> N * N correlation matrix (i.e. correlation of each startup to every other startup)

Once you have the correlation matrix over the "industry" fields, we use that to represent how much each startup relates
to every other startup.

## #Enjoy