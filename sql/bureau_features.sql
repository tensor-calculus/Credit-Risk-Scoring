-- bureau_features.sql
-- Aggregated credit bureau features per SK_ID_CURR
-- Source tables: bureau, bureau_balance

WITH bureau_agg AS (
    SELECT
        SK_ID_CURR,
        COUNT(*)                                                    AS BUREAU_CREDIT_COUNT,
        COUNT(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 END)        AS BUREAU_ACTIVE_COUNT,
        COUNT(CASE WHEN CREDIT_ACTIVE = 'Closed' THEN 1 END)        AS BUREAU_CLOSED_COUNT,
        CAST(COUNT(CASE WHEN CREDIT_ACTIVE = 'Active' THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0)                                   AS BUREAU_ACTIVE_RATIO,
        AVG(AMT_CREDIT_SUM)                                         AS BUREAU_AVG_CREDIT_SUM,
        MAX(AMT_CREDIT_SUM)                                         AS BUREAU_MAX_CREDIT_SUM,
        AVG(AMT_CREDIT_SUM_DEBT)                                    AS BUREAU_AVG_CREDIT_DEBT,
        MAX(AMT_CREDIT_MAX_OVERDUE)                                 AS BUREAU_MAX_OVERDUE,
        AVG(DAYS_CREDIT)                                            AS BUREAU_AVG_DAYS_CREDIT,
        MAX(CREDIT_DAY_OVERDUE)                                     AS BUREAU_CREDIT_DAY_OVERDUE_MAX,
        AVG(DAYS_CREDIT_ENDDATE - DAYS_CREDIT)                      AS BUREAU_AVG_CREDIT_DURATION,
        COALESCE(SUM(CNT_CREDIT_PROLONG), 0)                        AS BUREAU_PROLONG_COUNT
    FROM bureau
    GROUP BY SK_ID_CURR
),

-- Bureau balance: monthly status aggregations per SK_ID_CURR
-- STATUS values: C=closed, 0=no DPD, 1=1-30 DPD, 2=31-60, 3=61-90, 4=91-120, 5=120+, X=unknown
bureau_bal_agg AS (
    SELECT
        b.SK_ID_CURR,
        COUNT(CASE WHEN bb.STATUS IN ('1','2','3','4','5') THEN 1 END) AS BUREAU_BAL_DPD_MONTHS,
        MAX(CASE
            WHEN bb.STATUS IN ('1','2','3','4','5') THEN CAST(bb.STATUS AS INTEGER)
            ELSE 0
        END)                                                            AS BUREAU_BAL_MAX_DPD,
        CAST(COUNT(CASE WHEN bb.STATUS = 'C' THEN 1 END) AS DOUBLE)
            / NULLIF(COUNT(*), 0)                                       AS BUREAU_BAL_STATUS_C_RATIO
    FROM bureau b
    INNER JOIN bureau_balance bb ON b.SK_ID_BUREAU = bb.SK_ID_BUREAU
    GROUP BY b.SK_ID_CURR
)

SELECT
    ba.SK_ID_CURR,
    ba.BUREAU_CREDIT_COUNT,
    ba.BUREAU_ACTIVE_COUNT,
    ba.BUREAU_CLOSED_COUNT,
    ba.BUREAU_ACTIVE_RATIO,
    ba.BUREAU_AVG_CREDIT_SUM,
    ba.BUREAU_MAX_CREDIT_SUM,
    ba.BUREAU_AVG_CREDIT_DEBT,
    ba.BUREAU_MAX_OVERDUE,
    ba.BUREAU_AVG_DAYS_CREDIT,
    ba.BUREAU_CREDIT_DAY_OVERDUE_MAX,
    ba.BUREAU_AVG_CREDIT_DURATION,
    ba.BUREAU_PROLONG_COUNT,
    COALESCE(bba.BUREAU_BAL_DPD_MONTHS, 0)    AS BUREAU_BAL_DPD_MONTHS,
    COALESCE(bba.BUREAU_BAL_MAX_DPD, 0)        AS BUREAU_BAL_MAX_DPD,
    COALESCE(bba.BUREAU_BAL_STATUS_C_RATIO, 0) AS BUREAU_BAL_STATUS_C_RATIO
FROM bureau_agg ba
LEFT JOIN bureau_bal_agg bba ON ba.SK_ID_CURR = bba.SK_ID_CURR
