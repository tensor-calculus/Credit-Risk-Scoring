-- pos_cash_features.sql
-- Aggregated POS/Cash loan balance features per SK_ID_CURR
-- Source table: pos_cash_balance
-- ============================================================

SELECT
    SK_ID_CURR,
    COUNT(*)                                                        AS POS_COUNT,
    COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Active' THEN 1 END)     AS POS_ACTIVE_COUNT,
    COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Completed' THEN 1 END)  AS POS_COMPLETED_COUNT,
    MAX(SK_DPD)                                                     AS POS_MAX_DPD,
    AVG(CAST(SK_DPD AS DOUBLE))                                     AS POS_AVG_DPD,
    COUNT(CASE WHEN SK_DPD > 0 THEN 1 END)                          AS POS_DPD_MONTHS,
    CAST(COUNT(CASE WHEN SK_DPD > 0 THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0)                                       AS POS_DPD_RATIO,
    MAX(SK_DPD_DEF)                                                 AS POS_MAX_DPD_DEF,
    AVG(CNT_INSTALMENT_FUTURE)                                      AS POS_AVG_INSTALMENT_FUTURE
FROM pos_cash_balance
GROUP BY SK_ID_CURR
