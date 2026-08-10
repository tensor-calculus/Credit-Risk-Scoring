-- previous_app_features.sql
-- Aggregated previous application features per SK_ID_CURR
-- Source table: previous_application
-- ============================================================

SELECT
    SK_ID_CURR,
    COUNT(*)                                                          AS PREV_APP_COUNT,
    COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 END)     AS PREV_APPROVED_COUNT,
    COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Refused' THEN 1 END)      AS PREV_REFUSED_COUNT,
    CAST(COUNT(CASE WHEN NAME_CONTRACT_STATUS = 'Approved' THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0)                                         AS PREV_APPROVAL_RATE,
    AVG(AMT_APPLICATION)                                              AS PREV_AVG_AMT_APPLICATION,
    AVG(AMT_CREDIT)                                                   AS PREV_AVG_AMT_CREDIT,
    AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))                      AS PREV_AVG_CREDIT_TO_APP_RATIO,
    AVG(AMT_DOWN_PAYMENT)                                             AS PREV_AVG_DOWN_PAYMENT,
    AVG(DAYS_DECISION)                                                AS PREV_AVG_DAYS_DECISION,
    COUNT(CASE WHEN NAME_CONTRACT_TYPE = 'Cash loans' THEN 1 END)     AS PREV_CASH_COUNT,
    COUNT(CASE WHEN NAME_CONTRACT_TYPE = 'Consumer loans' THEN 1 END) AS PREV_CONSUMER_COUNT,
    AVG(CNT_PAYMENT)                                                  AS PREV_AVG_CNT_PAYMENT,
    CAST(COUNT(CASE WHEN NAME_CLIENT_TYPE = 'Repeater' THEN 1 END) AS DOUBLE)
        / NULLIF(COUNT(*), 0)                                         AS PREV_REPEATER_RATIO
FROM previous_application
GROUP BY SK_ID_CURR
