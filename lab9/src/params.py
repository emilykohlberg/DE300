# NHL seasons
seasons = range(2021, 2022)
# maximum number of games for a season
maxGames = 1500

# first part of url
url1 = "http://statsapi.web.nhl.com/api/v1/game/"
# second part of url
url2 = "/feed/live/"

# the name of the logging file
logFile = "nhl.log"
import logging
logLevel = logging.INFO

# mongo connection string
mongoConnectionString = "//johndoe:de300@localhost:27017"
# name of the mongo database
dbName = 'sports-ai'
# collection names in dbName
collectionNames = dict(
    rawpbp="nhlfeed",
    games="games",
    pbps="pbps" # play by play
)
# batch size to insert to mongo
mongoInsertBatchSize = 100
