############################################################
#
# File Name:    main.py
#
# Author:       Nick Klabjan
#
# Description:  Creates an NHL object which allows to make
#               directories and ultimately runs the program.
#
############################################################

import src.nhl_etl as ne
import src.params
import logging

logging.basicConfig(filename=src.params.logFile,level=src.params.logLevel,format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')

def main():
    nhl = ne.NHL_ETL()  # creates NHL object
    nhl.transferToMongoDB(incremental = True)  # downloads files from the NHL server; create also games and pbps collections
    nhl.fetch()

main()

