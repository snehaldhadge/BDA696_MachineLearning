SHOW ERRORS;
SHOW WARNINGS;
COMMIT;

USE baseball;

DROP TABLE  IF EXISTS allbatterdata;

CREATE TABLE allbatterdata as
(SELECT hit,atbat,batter,bc.game_id,Date(local_date) as l_dt from batter_counts bc
JOIN game g ON g.game_id = bc.game_id
ORDER BY game_id ,local_date );

DROP TABLE IF EXISTS rolling_average;

CREATE TABLE rolling_average  as
select  IFNULL((SUM(IFNULL(hist.Hit,0))/NULLIF(sum(IFNULL(hist.atbat,0)),0)),0) as Rolling_average,
       cur.batter,cur.game_id,cur.l_dt as Local_Date
from allbatterdata cur JOIN allbatterdata hist on
    cur.batter = hist.batter and hist.l_dt BETWEEN DATE_ADD(cur.l_dt, INTERVAL -100 day)
AND DATE_SUB(cur.l_dt,INTERVAL 1 DAY)
GROUP BY cur.batter,cur.game_id,cur.l_dt
ORDER BY cur.game_id;
