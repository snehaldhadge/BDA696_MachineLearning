	
##############################################################################################
# Assignment: 2
# filename:	Snehal_Assignment2.sql
# Notes:	MYSQL script for calculating Historic, Annual and rolling batting Average
# created: 	2020-09-15	
##############################################################################################

	
SHOW ERRORS;
SHOW WARNINGS;
COMMIT;

USE baseball;

###################################################################################################
# Historic Average
###################################################################################################

DROP TABLE IF EXISTS historic_average;

CREATE TABLE historic_average AS
SELECT IFNULL((SUM(hit))/NULLIF(SUM(atbat),0),0) as batting_average,batter
FROM batter_counts 
GROUP BY batter 
ORDER BY batting_average desc;

SELECT * FROM historic_average;

###################################################################################################
# Annual Average
###################################################################################################

DROP TABLE IF EXISTS annual_average;

CREATE TABLE annual_average AS
SELECT batter,year(g.local_date) as game_year,IFNULL(sum(Hit)/NULLIF(sum(atBat),0),0) as batting_average
FROM batter_counts bc 
JOIN game g on g.game_id  = bc.game_id 
GROUP BY  batter,game_year;

SELECT * FROM annual_average;

###################################################################################################
# Rolling
# Creating View vw_allbatterdata for combining batters_count with local_date and ordering by game 
# date 
###################################################################################################

DROP VIEW IF EXISTS vw_allbatterdata;

CREATE VIEW vw_allbatterdata as
(SELECT hit,atbat,batter,bc.game_id,Date(local_date) as l_dt from batter_counts bc 
JOIN game g ON g.game_id = bc.game_id 
ORDER BY game_id ,local_date )

DROP TABLE IF EXISTS rolling_average;

CREATE TABLE rolling_average  as
select IFNULL((SELECT SUM(IFNULL(Hit,0))/NULLIF(sum(IFNULL(atbat,0)),0) from vw_allbatterdata vw1 where vw1.l_dt BETWEEN DATE_ADD(vw.l_dt, INTERVAL -100 day)
AND DATE_SUB(vw.l_dt,INTERVAL 1 DAY) and vw1.batter = vw.batter),0) as Rolling_average , vw.batter,vw.game_id,vw.l_dt 
from vw_allbatterdata vw 
ORDER BY vw.game_id;


###################################################################################################
# SELECT rolling average for all players for a Game
###################################################################################################

SELECT * from rolling_average  order by game_id ;


###################################################################################################
# SELECT rolling average for all players for a Game on a given date
###################################################################################################

SELECT * from rolling_average where l_dt = '2009-05-07';


