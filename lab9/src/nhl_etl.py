import urllib.error, urllib.request
import json
import src.params
import pymongo
import logging
import bisect

"""
Creates an NHL object that allows one to 
create directories and download NHL files.
"""
class NHL_ETL:

    """
    Constructs new NHL_ETL object; set mongoDB; no cleaning to the db
    """
    def __init__(self):
        client = pymongo.MongoClient('mongodb:' + src.params.mongoConnectionString)
        self.db = getattr(client, src.params.dbName) # also be client[src.params.dbName]
        self.rawpbp = getattr(self.db, src.params.collectionNames['rawpbp'])
        self.pbp = getattr(self.db, src.params.collectionNames['pbps'])
        self.games = getattr(self.db, src.params.collectionNames['games']) # can also be db["games"]

    """
    Downloads .json files from the NHL server.
    Incremental works only for a single season
    """
    def transferToMongoDB(self, incremental):
        """
        :param incremental: if false, don't check if game already in mongo; if true, check if game already in mongo
        :return: nothing
        """
        logging.info("Starting to download games and storing to mongo ...")
        allGames = []
        if incremental == True:
            if len(src.params.seasons) != 1:
                logging.info("Incremental loading with more than one season does not make sense! Aboarting!")
                return
            allGames = [v['gamePk'] for v in list(self.rawpbp.find({}, {"gamePk": 1, "_id": 0}))]
            # sort since we are going to do binary search
            allGames.sort()

        # we insert in batches to mongo
        insertGameBatch = []
        insertPbPBatch = []
        insertRawBatch = []

        # traverses through NHL seasons
        for x in src.params.seasons:
            # counter to count number of files downloaded
            counter = 0
            # traverses through each game in a season
            for y in range(0, src.params.maxGames):
                try:
                    # saves URL address that contains game's data to variable 'url'
                    url = src.params.url1 + str(x) + "02" + str("%04d" % y) + src.params.url2
                    # grabs the game's data from the url
                    jsonString = urllib.request.urlopen(url).read().decode()
                    data = json.loads(jsonString)
                    # make sure the games has already played; applies really only to the ongoing season
                    if data['gameData']['status']['detailedState'] == "Final":
                        # add data to mongo
                        insert = True
                        if incremental == True and allGames:
                            # check if game already in mongo by binary search
                            insertPoint = bisect.bisect(allGames, data['gamePk'])
                            # assumption: the new game would not be the first one
                            isPresent = allGames[insertPoint - 1] == data['gamePk']
                            if isPresent == True:
                                insert = False
                        if insert == True:
                            # game not already in mongo; insert it
                            insertRawBatch.append(data)
                            counter = counter + 1
                            if len(insertRawBatch) % src.params.mongoInsertBatchSize == 0:
                                self.rawpbp.insert_many(insertRawBatch)
                                insertRawBatch.clear()
                            # add to the games collection
                            valuesGames = self._extractGameDataFromRawRecord(data)
                            insertGameBatch.append(valuesGames)
                            # insert records if we have enough of them
                            if len(insertRawBatch) == 0:
                                self.games.insert_many(insertGameBatch)
                                insertGameBatch.clear()
                            # add to the pbp collection
                            valuesPbP = self._extractPbPDataFromRawRecord(data)
                            insertPbPBatch += valuesPbP
                            # insert records if we have already enough of them
                            if len(insertRawBatch) == 0:
                                self.pbp.insert_many(insertPbPBatch)
                                insertPbPBatch.clear()

                # throws exception if url doesn't exist (ie no NHL  game)
                except urllib.error.HTTPError:
                    pass
            # prints out number of files downloaded for each season
            logging.info("Season " + str(x) + ": " + str(counter) + " games downloaded and stored to mongo")

        # insert the last batch
        if len(insertRawBatch) > 0:
            self.rawpbp.insert_many(insertRawBatch)
        if len(insertGameBatch) > 0:
            self.games.insert_many(insertGameBatch)
        if len(insertPbPBatch) > 0:
            self.pbp.insert_many(insertPbPBatch)

        # set gamePk as index
        self.rawpbp.create_index([('gamePk', pymongo.ASCENDING)],unique = True)
        self.games.create_index([('gamePk', pymongo.ASCENDING)], unique=True)

    """
    from a raw record extract game data
    """
    def _extractGameDataFromRawRecord(self,game):
        """
        :param game: one raw record
        :return: dictionary to be inserted to mongo
        """

        values = dict()
        values['gamePk'] = game["gamePk"]
        values['season'] = game["gameData"]["game"]["season"]
        values['dateTime'] = game["gameData"]["datetime"]["dateTime"]
        values['awayTeam'] = game["gameData"]["teams"]["away"]["name"]
        values['homeTeam'] = game["gameData"]["teams"]["home"]["name"]
        # players is dictionary that holds all data about players; key is the player id in json (specified by NHL)
        players = dict()
        # game related data about players
        for k, v in game["gameData"]["players"].items():
            if 'rosterStatus' in v:
                currentTeam = None
                if v["rosterStatus"] == 'Y' and v["active"] == True:
                    currentTeam = v["currentTeam"]["name"]
                players[str(k)] = {"fullName": v["fullName"], "rosterStatus": v["rosterStatus"], "active": v["active"],
                            "currentTeamName": currentTeam}

        # boxscore data for each player; away and home are used in json with different keys
        def populateBoxscore(team):
            for k, v in game["liveData"]["boxscore"]["teams"][team]["players"].items():
                if str(k) in players:
                    if "skaterStats" in v["stats"]:
                        players[str(k)].update({"timeOnIce": v["stats"]["skaterStats"]["timeOnIce"],
                                        "powerPlayTimeOnIce": v["stats"]["skaterStats"]["powerPlayTimeOnIce"],
                                        "shortHandedTimeOnIce": v["stats"]["skaterStats"]["shortHandedTimeOnIce"]})
                    else:
                        players[str(k)].update({"timeOnIce": None, "powerPlayTimeOnIce": None, "shortHandedTimeOnIce": None})
                    values["players"] = players

        populateBoxscore("away")
        populateBoxscore("home")

        return values

    """
     from a raw record extract pbp data
     """

    def _extractPbPDataFromRawRecord(self, game):
        """
        :param game: one raw record
        :return: list of dictionaries to be inserted to mongo; there is one entry in the list per play
        """

        allValues = []
        for v in game["liveData"]["plays"]["allPlays"]:
            values = {'x': None, 'y': None, "team": None, "players": {}}
            values['gamePk'] = game["gamePk"]
            values["event"] = v["result"]["event"]
            values["description"] = v["result"]["description"]
            values["period"] = v["about"]["period"]
            values["periodTime"] = v["about"]["periodTime"]
            values["goalsAway"] = v["about"]["goals"]["away"]
            values["goalsHome"] = v["about"]["goals"]["home"]
            if 'x' in v["coordinates"]:
                values["x"] = v["coordinates"]["x"]
            if 'y' in v['coordinates']:
                values["y"] = v["coordinates"]["y"]
            if "team" in v:
                values["team"] = v["team"]["name"]
            # players involved in action
            if 'players' in v:
                players = {str(u['player']['id']): {'fullName':u['player']['fullName'],'playerType':u['playerType']} for u in v['players']}
                values['players'] = players
            allValues.append(values)

        return allValues

    def fetch(self):
        collections = self.db.list_collection_names()
        print (f"collections: {collections}")
        
        values = self.games.find().skip(5).limit(8)
        for v in values:
            print(v)

        print("New query")

        # the second parameter is what fields to return (basic concept: key: 0/1 where 0 not to return and 1 to return
        # the first parameter is the filter
        values = self.games.find({"awayTeam": "Chicago Blackhawks", "homeTeam": "New York Islanders"}, {'dateTime': 1, "awayTeam": 1, "homeTeam": 1})
        for v in values:
            print(v)
        