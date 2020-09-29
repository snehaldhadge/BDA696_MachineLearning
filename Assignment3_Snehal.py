#!/usr/bin/env python3
import sys

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from BattingAvgCalc import BattingAvgCalc


# Printing heading before each section
def print_heading(title):
    print("*" * 80)
    print(title)
    print("*" * 80)
    return


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()
    b_counts = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://127.0.0.1:3306/baseball")
        .option("driver", "org.mariadb.jdbc.Driver")
        .option(
            "dbtable",
            "(select Hit,Atbat,batter,bc.game_id,DATE(g.local_date) as l_dt \
           FROM batter_counts bc join game g on g.game_id=bc.game_id)batter_counts_all",
        )
        .option("user", "root")
        .option("password", "root")
        .load()
    )
    b_counts.createOrReplaceTempView("batter_counts_all")
    b_counts.persist(StorageLevel.DISK_ONLY)
    results = spark.sql(
        """SELECT SUM(bc_i.Hit) as Sum_H,SUM(bc_i.atbat) as Sum_A,bc_o.batter,bc_o.game_id \
                FROM batter_counts_all bc_o JOIN \
                batter_counts_all bc_i ON bc_i.batter = bc_o.batter AND \
                bc_i.l_dt BETWEEN DATE_SUB(bc_o.l_dt,100) AND DATE_SUB(bc_o.l_dt,1)  \
                GROUP BY bc_o.batter, bc_o.game_id \
                ORDER BY bc_o.game_id """
    )
    batting_avg_cal = BattingAvgCalc(
        inputCols=["Sum_H", "Sum_A"], outputCol="rolling_average"
    )
    rolling_avg = batting_avg_cal.transform(results)
    print_heading("Rolling Average Calculation results")
    rolling_avg.show()
    print_heading("100 day Rolling Average for each batter for given game_id")
    rolling_avg.filter(rolling_avg.game_id == 5).select(
        ["batter", "game_id", "rolling_average"]
    ).show()


if __name__ == "__main__":
    sys.exit(main())
