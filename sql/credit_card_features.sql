-- credit_card_features.sql
-- Aggregated credit card balance features per SK_ID_CURR
-- Source table: credit_card_balance

SELECT
    SK_ID_CURR,
    COUNT(*)                                                         AS CC_COUNT,
    AVG(AMT_BALANCE)                                                 AS CC_AVG_BALANCE,
    MAX(AMT_BALANCE)                                                 AS CC_MAX_BALANCE,
    AVG(CAST(AMT_CREDIT_LIMIT_ACTUAL AS DOUBLE))                     AS CC_AVG_CREDIT_LIMIT,
    AVG(AMT_BALANCE / NULLIF(CAST(AMT_CREDIT_LIMIT_ACTUAL AS DOUBLE), 0))
                                                                     AS CC_AVG_UTILIZATION,
    MAX(AMT_BALANCE / NULLIF(CAST(AMT_CREDIT_LIMIT_ACTUAL AS DOUBLE), 0))
                                                                     AS CC_MAX_UTILIZATION,
    AVG(AMT_DRAWINGS_CURRENT)                                        AS CC_AVG_DRAWINGS,
    AVG(AMT_DRAWINGS_ATM_CURRENT)                                    AS CC_AVG_ATM_DRAWINGS,
    AVG(AMT_PAYMENT_CURRENT)                                         AS CC_AVG_PAYMENT,
    MAX(SK_DPD)                                                      AS CC_MAX_DPD,
    COUNT(CASE WHEN SK_DPD > 0 THEN 1 END)                           AS CC_DPD_MONTHS,
    AVG(AMT_INST_MIN_REGULARITY)                                     AS CC_AVG_MIN_INSTALMENT
FROM credit_card_balance
GROUP BY SK_ID_CURR
